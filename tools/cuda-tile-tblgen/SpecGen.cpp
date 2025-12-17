//===- SpecGen.cpp - MLIR operation documentation generator -----*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Uses the description of operations to generate documentation.
//
//===----------------------------------------------------------------------===//

#include "SpecGen.h"

#include "mlir/Support/IndentedOstream.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Property.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "CudaTileAttr.h"
#include "CudaTileOp.h"
#include "Emitter.h"
#include <sstream>
#include <string>
#include <variant>

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;
using namespace cudatile::tblgen;
using mlir::tblgen::Operator;

// Helper type to make it cleaner to write visitor for std::variant.
template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// The path to the file containing the pre-written header text for each section.
static const char *OP_CLASS_HEADING_PATH = "/sections/op_class_headings/";

static cl::OptionCategory opDefGenCat("Options for op definition generators");

static cl::opt<std::string> opIncFilter(
    "op-include-regex",
    cl::desc("Regex of name of op's to include (no filter if empty)"),
    cl::cat(opDefGenCat));
static cl::opt<std::string> opExcFilter(
    "op-exclude-regex",
    cl::desc("Regex of name of op's to exclude (no filter if empty)"),
    cl::cat(opDefGenCat));

static std::string getOperationName(const Record &def) {
  auto prefix = def.getValueAsDef("opDialect")->getValueAsString("name");
  auto opName = def.getValueAsString("opName");
  if (prefix.empty())
    return std::string(opName);
  return std::string(formatv("{0}.{1}", prefix, opName));
}

static std::vector<const Record *>
getRequestedOpDefinitions(const RecordKeeper &records) {
  const Record *classDef = records.getClass("Op");
  if (!classDef)
    PrintFatalError("ERROR: Couldn't find the 'Op' class!\n");

  Regex includeRegex(opIncFilter), excludeRegex(opExcFilter);
  std::vector<const Record *> defs;
  for (const auto &def : records.getDefs()) {
    if (!def.second->isSubClassOf(classDef))
      continue;
    // Include if no include filter or include filter matches.
    if (!opIncFilter.empty() &&
        !includeRegex.match(getOperationName(*def.second)))
      continue;
    // Unless there is an exclude filter and it matches.
    if (!opExcFilter.empty() &&
        excludeRegex.match(getOperationName(*def.second)))
      continue;
    defs.push_back(def.second.get());
  }

  return defs;
}

static FormattedExample processExample(const std::string &example) {
  std::vector<std::tuple<int, int>> lineRanges;
  std::stringstream exampleReindented;
  int reindent = INT_MAX;
  int dedent = INT_MAX;

  std::vector<std::string> lines;
  std::string line;
  std::istringstream stream(example);

  int lineNumber = 1;
  while (std::getline(stream, line)) {
    // Find first non-whitespace character
    size_t firstNonWhitespace = line.find_first_not_of(" \t");
    if (firstNonWhitespace != std::string::npos) {
      // If line starts with #, update reindent if needed
      int leadingSpaces = firstNonWhitespace;
      if (line[firstNonWhitespace] == '#') {
        size_t firstNonWSAfterHash =
            line.find_first_not_of(" \t", firstNonWhitespace + 1);
        // We want to remove the leading # and the leading spaces but preserve
        // the rest of the whitespace as we want normalize the whitespace.
        std::string lineWithoutHash = line.substr(0, firstNonWhitespace) +
                                      line.substr(firstNonWhitespace + 1);
        lines.push_back(lineWithoutHash);
        leadingSpaces = firstNonWSAfterHash - 1;
      } else {
        lines.push_back(line);
        lineRanges.emplace_back(lineNumber, lineNumber);
        dedent = std::min(dedent, leadingSpaces);
      }
      // Compute how must to reindent the line by.
      reindent = std::min(reindent, leadingSpaces);
    } else {
      lines.push_back("");
      lineRanges.emplace_back(lineNumber, lineNumber);
    }
    lineNumber++;
  }

  // If there was no leading indentation we don't want to reindent
  // we used INT_MAX as a sentinel value.
  reindent = std::max(0, reindent);
  // We want to dedent the lines by the max of the visible lines's leading
  // whitespace.
  //
  // For example if we display the body of a function we will reindent
  // correctly but when we render the lines they will all have the same
  // leading whitespace.
  dedent = std::max(0, dedent);
  dedent -= reindent;

  // Before we tracked only one line spans (i.e., 1-1, 2-2)
  // this compresses continous spans (i.e., 1-2) to reduce the generated
  // noise.
  std::vector<std::tuple<int, int>> compressedLineRanges;
  int startRange = 1;
  int endRange = 0;
  for (const auto &lineRange : lineRanges) {
    if (std::get<0>(lineRange) == endRange + 1) {
      endRange++;
    } else {
      compressedLineRanges.emplace_back(startRange, endRange);
      startRange = std::get<0>(lineRange);
      endRange = std::get<1>(lineRange);
    }
  }

  // Make sure to add the last range in case the last range
  // has no breaks.
  compressedLineRanges.emplace_back(startRange, endRange);

  for (const auto &line : lines) {
    if (!line.empty()) {
      // std::cout << "line: " << line << std::endl;
      exampleReindented << line.substr(reindent) << std::endl;
    } else {
      exampleReindented << std::endl;
    }
  }

  return {compressedLineRanges, exampleReindented.str(), dedent};
}

void emitSummary(StringRef summary, raw_ostream &os) {
  if (!summary.empty()) {
    StringRef trimmed = summary.trim();
    char first = std::toupper(trimmed.front());
    StringRef rest = trimmed.drop_front();
    os << "\n_" << first << rest << "_\n\n";
  }
}

/// Emit the given named constraint.
template <typename T>
static void emitNamedConstraint(const T &it, raw_ostream &os) {
  if (!it.name.empty())
    os << "| `" << it.name << "`";
  else
    os << "&laquo;unnamed&raquo;";
  os << " | " << it.constraint.getSummary() << "\n";
}

std::vector<std::string> covertSyntaxToSignature(const Operator &op) {
  std::vector<std::string> signature;

  std::string opName = op.getOperationName();
  StringRef format = op.getAssemblyFormat().trim();

  // Split the string by spaces.
  SmallVector<StringRef, 8> split;
  format.split(split, ' ');

  for (auto &parameter : split) {
    if (parameter.trim(" \n") == "`:`") {
      break;
    } else if (parameter.trim(" \n") == "attr-dict") {
      continue;
    }
    signature.push_back(parameter.trim("`").str());
  }

  return signature;
}

static void emitAllTypesMatch(SpecEmitter &emitter, const AllTypesMatch &arg) {
  int i = 0;
  for (auto &name : arg.names) {
    if (i > 0) {
      emitter.os << (i == static_cast<int>(arg.names.size()) - 1 ? " and "
                                                                 : ", ");
    }
    emitter.os << ":code:`" << name << "`";
    i++;
  }
  emitter.os << " must have the same shape and element type ("
             << TileIRTy{arg.type} << ").\n";
}

static void emitAllElementTypeMatch(SpecEmitter &emitter,
                                    const AllElementTypeMatch &arg) {
  int i = 0;
  for (auto &name : arg.names) {
    if (i > 0) {
      if (i == static_cast<int>(arg.names.size()) - 1) {
        emitter.os << " and ";
      } else {
        emitter.os << ", ";
      }
    }
    emitter.os << Code{name};
    i++;
  }
  emitter.os << " must have the same element type (" << TileIRTy{arg.type}
             << ").\n";
}

static void emitAnyTypeOf(SpecEmitter &emitter, const AnyTypeOf &arg) {
  emitter.os << "The type of ";
  int i = 0;
  for (auto &name : arg.names) {
    if (i > 0) {
      emitter.os << ", ";
    }
    emitter.os << Code{name};
    i++;
  }
  emitter.os << " must be one of the allowed types.\n";
}

static void emitAllRanksMatch(SpecEmitter &emitter, const AllRanksMatch &arg) {
  int i = 0;
  for (auto &name : arg.names) {
    if (i > 0) {
      if (i == static_cast<int>(arg.names.size()) - 1) {
        emitter.os << " and ";
      } else {
        emitter.os << ", ";
      }
    }
    emitter.os << Code{name};
    i++;
  }
  emitter.os << " must have the same rank.\n";
}

static void emitTypesMatchWith(SpecEmitter &emitter,
                               const TypesMatchWith &arg) {
  emitter.os << arg.message << "\n";
}

static void emitSameTypeOperands(SpecEmitter &emitter,
                                 const SameTypeOperands &arg) {
  int i = 0;
  for (auto &name : arg.names) {
    if (i > 0) {
      if (i == static_cast<int>(arg.names.size()) - 1) {
        emitter.os << " and ";
      } else {
        emitter.os << ", ";
      }
    }
    emitter.os << Code{name};
    i++;
  }
  emitter.os << " must have the same shape and element type ("
             << TileIRTy{arg.type} << ").\n";
}

static void
emitSameOperandsAndResultShape(SpecEmitter &emitter,
                               const SameOperandsAndResultShape &arg) {
  int i = 0;
  for (auto &name : arg.names) {
    if (i > 0) {
      if (i == static_cast<int>(arg.names.size()) - 1) {
        emitter.os << " and ";
      } else {
        emitter.os << ", ";
      }
    }
    emitter.os << Code{name};
    i++;
  }
  emitter.os << " must have the same shape.\n";
}

static void emitSameOperandsAndResultElementType(
    SpecEmitter &emitter, const SameOperandsAndResultElementType &arg) {
  int i = 0;
  for (auto &name : arg.names) {
    if (i > 0) {
      emitter.os << (i == static_cast<int>(arg.names.size()) - 1 ? " and "
                                                                 : ", ");
    }
    emitter.os << Code{name};
    i++;
  }
  emitter.os << " must have the same element type (" << arg.type << ").\n";
}

static void emitOperationTrait(SpecEmitter &emitter,
                               const OperationTrait &arg) {
  emitter.os << arg.description << "\n";
}

static void emitOperationConstraint(SpecEmitter &emitter,
                                    const OperationConstraint &constraint) {
  std::visit(
      overloaded{
          [&](const AllTypesMatch &arg) { emitAllTypesMatch(emitter, arg); },
          [&](const AllElementTypeMatch &arg) {
            emitAllElementTypeMatch(emitter, arg);
          },
          [&](const AnyTypeOf &arg) { emitAnyTypeOf(emitter, arg); },
          [&](const AllRanksMatch &arg) { emitAllRanksMatch(emitter, arg); },
          [&](const TypesMatchWith &arg) { emitTypesMatchWith(emitter, arg); },
          [&](const SameTypeOperands &arg) {
            emitSameTypeOperands(emitter, arg);
          },
          [&](const SameOperandsAndResultShape &arg) {
            emitSameOperandsAndResultShape(emitter, arg);
          },
          [&](const SameOperandsAndResultElementType &arg) {
            emitSameOperandsAndResultElementType(emitter, arg);
          },
          [&](const OperationTrait &arg) { emitOperationTrait(emitter, arg); }},
      constraint);
}

static void emitEnumAttribute(SpecEmitter &emitter,
                              const TileIREnumAttr &enumAttr) {
  emitter.emitAnchor(enumAttr.getAnchor());

  emitter.emitDescription(enumAttr.prefixDescription);

  std::vector<std::string> disabledVariants;
  for (const auto &enumcase : enumAttr.enumcases) {
    if (!enumcase.isSelected) {
      disabledVariants.push_back(Code{enumcase.str});
      continue;
    }

    emitter.os << "- " << Code{enumcase.str} << " - " << enumcase.description
               << "\n";
  }

  emitter.os << "\n\n";

  if (!disabledVariants.empty()) {
    emitter.os
        << "Note: The following variants are not supported by this operation: "
        << llvm::join(disabledVariants, ", ") << ".\n";
  }

  emitter.os << "\n\n";
  emitter.emitDescription(enumAttr.suffixDescription);
}

static void emitAttributeDef(SpecEmitter &emitter,
                             const TileIRAttrDef &attrDef) {
  emitter.emitAnchor(attrDef.getAnchor());
  std::cout << "prefixDescription: " << attrDef.prefixDescription << "\n";
  emitter.emitDescription(attrDef.prefixDescription);
  int i = 0;
  for (auto &exampleText : attrDef.examples) {
    auto processedExample = processExample(exampleText);
    auto exampleName =
        attrDef.opName + "_example_" + attrDef.name + "_" + std::to_string(i);
    emitter.emitExample(exampleName, processedExample);
    i++;
  }
  emitter.os << "\n\n";
}

static void
emitAttributeInterface(SpecEmitter &emitter,
                       const TileIRAttrInterface &attrInterface,
                       const std::vector<const Record *> &attrDefs) {
  std::cout << "emitAttributeInterface: " << attrInterface.name << std::endl;
  emitter.emitAnchor(attrInterface.getAnchor());
  emitter.emitDescription(attrInterface.prefixDescription);

  auto implementators = findInterfaceImplementors(attrInterface, attrDefs);

  emitter.os << Header(OP_DETAILS_HEADER_LEVEL,
                       attrInterface.name + " Implementers");

  for (auto &attr : implementators) {
    emitter.os << Header(OP_DETAILS_HEADER_LEVEL, attr.name);
    emitAttributeDef(emitter, attr);
  }

  emitter.emitDescription(attrInterface.suffixDescription);
}

static void emitAttribute(SpecEmitter &emitter, const TileIRAttr &attr,
                          const std::vector<const Record *> &attrDefs) {
  std::visit(
      overloaded{
          [&](const TileIREnumAttr &arg) { emitEnumAttribute(emitter, arg); },
          [&](const TileIRAttrDef &arg) { emitAttributeDef(emitter, arg); },
          [&](const TileIRAttrInterface &arg) {
            emitAttributeInterface(emitter, arg, attrDefs);
          },
      },
      attr);
}

/// Emit the signature of an operation.
static void emitOperationSignature(SpecEmitter &emitter,
                                   const CudaTileOp &cuda_tile_op) {
  auto &signature = cuda_tile_op.signature;
  // if (op.hasAssemblyFormat()) {
  //   auto raw_signature = covertSyntaxToSignature(op);
  //   emitter.emitCodeBlock([&](raw_ostream &os) {
  //     os << signature.name << " ";
  //     for (auto &parameter : raw_signature) {
  //       os << parameter << " ";
  //     }
  //   });
  // } else {
  emitter.emitCodeBlock([&](raw_ostream &os) {
    os << signature.name << " ";
    for (auto &parameter : signature.parameters) {
      os << "%" << parameter.name << " ";
    }
  });
  //}

  emitter.os << Header(OP_DETAILS_HEADER_LEVEL, "Parameters");

  if (signature.parameters.empty()) {
    emitter.os << "No parameters.\n";
  }

  for (auto &parameter : signature.parameters) {
    emitter.os << "- **" << parameter.name << "**";
    emitter.os << " (" << parameter.getTypeDescription() << ")";
    emitter.os << " - " << parameter.getDescription();
    if (!parameter.sinceVersion.empty()) {
      emitter.os << " " << Badge::successLine(parameter.sinceVersion);
    }
    emitter.os << "\n";
  }

  emitter.os << "\n\n";

  emitter.os << Header(OP_DETAILS_HEADER_LEVEL, "Results");

  if (signature.results.empty()) {
    emitter.os << "No results.\n";
  }

  for (auto &parameter : signature.results) {
    // TODO: Figure out how to ignore "spelling errors" in code names.
    // Ignore spell checks on parameter/result names
    // emitter.os << "- :spelling:ignore:`**" << parameter.name << "**`";
    emitter.os << "- **" << parameter.name << "**";
    emitter.os << " (" << parameter.getTypeDescription() << ")";
    emitter.os << " - " << parameter.getDescription();

    if (!parameter.sinceVersion.empty()) {
      emitter.os << " " << Badge::successLine(parameter.sinceVersion);
    }

    emitter.os << "\n";
  }

  emitter.os << "\n\n";
}

static void emitOperationExample(SpecEmitter &emitter,
                                 const std::string &exampleName,
                                 const std::string &example) {
  auto processedExample = processExample(example);
  emitter.emitExample(exampleName, processedExample);
}

// Emit documentation for an operation of the rough form:
//
// OP_NAME
//
// SHORT_DESCRIPTION
//
// SIGNATURE
//
// ARGUMENTS
//
// RESULTS
//
// DESCRIPTION
//
// CONSTRAINTS
static void emitOpDoc(SpecEmitter &emitter, CudaTileOp &cudaTileOp,
                      std::vector<const Record *> &attrDefs) {
  // We can create per-operation badges that we can attach when rendering it.
  std::vector<Badge> badges;

  // TODO: get the operation version here, we need to pull OperationSignature
  // up.
  emitter.emitOpHeading(cudaTileOp.getOperationName());

  // TODO: This should probably be folded into an emitter method or
  // emitOpHeading.
  bool first = true;
  for (const auto &badge : badges) {
    if (first) {
      emitter.os << badge;
    } else {
      emitter.os << " " << badge;
    }
  }

  if (!badges.empty()) {
    emitter.os << "\n\n";
  }

  // Emit the summary, syntax, and description if present.
  if (cudaTileOp.op.hasSummary())
    emitter.emitSummary(cudaTileOp.op.getSummary());

  emitOperationSignature(emitter, cudaTileOp);

  emitter.os << Header(OP_DETAILS_HEADER_LEVEL, "Description");
  if (cudaTileOp.op.hasDescription())
    // todo delete this helper and move to emitter.h
    emitter.emitDescription(cudaTileOp.getDescription());

  // Emit the attributes.
  auto attributes = cudaTileOp.getAttributes();
  for (const auto &enumAttr : attributes) {
    emitAttribute(emitter, enumAttr, attrDefs);
  }

  // Emit the description tables.
  auto descriptionTables = cudaTileOp.getDescriptionTables();
  int i = 0;
  for (auto &table : descriptionTables) {
    auto anchor =
        "table-" + cudaTileOp.getOperationName() + "-" + std::to_string(i);
    table.anchors.push_back(anchor);
    emitter.os << table;
    i++;
  }

  emitter.os << "\n";

  // Finally emit the constraints.
  emitter.os << Header(OP_DETAILS_HEADER_LEVEL, "Constraints");
  emitter.os << "\n";
  if (!cudaTileOp.signature.constraints.empty()) {
    for (auto &constraint : cudaTileOp.signature.constraints) {
      emitter.os << "- ";
      emitOperationConstraint(emitter, constraint);
      emitter.os << "\n";
    }
  } else {
    emitter.os << "No constraints."
               << "\n\n";
  }

  emitter.os << "\n\n";

  if (!cudaTileOp.getMLIRExamples().empty()) {
    emitter.os << Header(OP_DETAILS_HEADER_LEVEL, "Examples");
    int i = 0;
    for (auto &example : cudaTileOp.getMLIRExamples()) {
      std::string exampleName =
          cudaTileOp.getOperationName() + "_" + std::to_string(i);
      emitOperationExample(emitter, exampleName, example);
      i++;
    }
  }

  // TODO: emit information about the regions.

  // Emit successors.
  // if (op.getNumSuccessors() != 0) {
  //   os << Header(OP_DETAILS_HEADER_LEVEL, "Successors:");
  //   os << "| Successor | Description |\n"
  //      << "| :-------: | ----------- |\n";
  //   for (const auto &it : op.getSuccessors())
  //     emitNamedConstraint(it, os);
  // }

  emitter.os << "\n";
}

// These are the declared sections.
static const char *const cudaTileSections[] = {
    "Core",           "Conversions",   "Control Flow", "Memory",
    "Floating Point", "Integer",       "Bitwise",      "Atomics",
    "Views",          "Miscellaneous", "Testing",
};

static const int NUMBER_SECTIONS =
    sizeof(cudaTileSections) / sizeof(cudaTileSections[0]);

static std::vector<std::pair<std::string, std::vector<const Record *>>>
splitBySections(const RecordKeeper &records) {
  // First we sort by `cudaTileGroup` then we emit.
  std::unordered_map<std::string, std::vector<const Record *>> groupedOps;

  auto opDefs = getRequestedOpDefinitions(records);

  for (const Record *opDef : opDefs) {
    CudaTileOp op = CudaTileOp(opDef);
    auto cudaTileGroup = op.getCudaTileSpecGroup();
    if (groupedOps.count(cudaTileGroup) == 0) {
      groupedOps[cudaTileGroup] = {opDef};
    } else {
      groupedOps[cudaTileGroup].push_back(opDef);
    }
  }

  std::unordered_map<std::string, int> cudaTileGroupLabels;
  for (int i = 0; i < NUMBER_SECTIONS; i++) {
    // std::cout << "LABEL";
    // std::cout << cudaTileSections[i] << " " << i + 1 << std::endl;
    cudaTileGroupLabels[cudaTileSections[i]] = i + 1;
  }

  std::vector<std::pair<std::string, std::vector<const Record *>>>
      orderedSections(groupedOps.begin(), groupedOps.end());

  std::sort(orderedSections.begin(), orderedSections.end(),
            [&](const std::pair<std::string, std::vector<const Record *>> &a,
                std::pair<std::string, std::vector<const Record *>> &b) {
              int aScore = -1;
              int bScore = -1;

              if (cudaTileGroupLabels.count(a.first) > 0) {
                aScore = cudaTileGroupLabels[a.first];
              }

              if (cudaTileGroupLabels.count(b.first) > 0) {
                bScore = cudaTileGroupLabels[b.first];
              }

              return aScore < bScore;
            });

  return orderedSections;
}

void cudatile::tblgen::generateSpec(
    raw_ostream &os, const llvm::RecordKeeper &records,
    const std::optional<std::string> &examplesDirectory) {
  raw_indented_ostream raw_ios(os);
  SpecEmitter emitter(raw_ios, examplesDirectory);

  // This should probably be moved to the emitter.
  emitter.emitComment(AUTO_GENERATED_MESSAGE);

  // The spec generation today only considers the dialect ops and nothing else.

  // Split the ops by sections.
  auto orderedSections = splitBySections(records);

  std::vector<const Record *> attrDefs;
  for (const auto &def : records.getDefs()) {
    if (def.second->isSubClassOf("AttrDef")) {
      attrDefs.push_back(def.second.get());
    }
  }

  for (auto const &pair : orderedSections) {
    // The first part of the pair is the section name/heading.
    auto cudaTileGroupLabel = pair.first;
    // The second is a lit of the records corresponding to the operations in the
    // section/group.
    auto groupOps = pair.second;

    // Skip documenting any testing operations.
    if (cudaTileGroupLabel == "Testing") {
      continue;
    }

    // An anchor declares a thing that can be references elsewhere in the
    // document.
    //
    // Generate an anchor of the form op-group-<cudaTileGroupLabel>.
    std::string normalizedGroupLabel(cudaTileGroupLabel);
    std::replace(normalizedGroupLabel.begin(), normalizedGroupLabel.end(), ' ',
                 '-');
    emitter.emitAnchor("op-group", normalizedGroupLabel);

    // Emit a header for the section at the SECTION_HEADER_LEVEL.
    //
    // Generates:
    //
    // <cudaTileGroupLabel>
    // ====================

    emitter.os << Header(SECTION_HEADER_LEVEL, cudaTileGroupLabel);

    // Include the pre-written header text for the section.
    //
    // Generates:
    //
    // .. include:: /sections/op_class_headings/<cudaTileGroupLabel>_heading.rst
    //
    // The is the pre-written text for the section.
    //
    // TODO: modify to use emitInclude.
    emitter.os << ".. include:: " << OP_CLASS_HEADING_PATH << cudaTileGroupLabel
               << "_heading.rst"
               << "\n\n";

    // Finally we iterate over each operation in the group and emit a section
    // for it.
    for (auto opDef : groupOps) {
      // Note: construct here due to ownership/lifetime issues with storing
      // the ops in vector.
      Operator op(opDef);
      CudaTileOp cudaTileOp(op);
      // Call emitOpDoc with the emitter and the operation.
      emitOpDoc(emitter, cudaTileOp, attrDefs);
    }

  }
}
