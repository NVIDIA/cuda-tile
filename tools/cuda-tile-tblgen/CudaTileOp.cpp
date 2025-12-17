//===- CudaTileOp.cpp - CUDA Tile operation definitions ---------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file implements the CUDA Tile dialect operations.
//
//===----------------------------------------------------------------------===//

#include "CudaTileOp.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"

#include <unordered_map>
#include <unordered_set>

using namespace llvm;
using namespace mlir;

namespace cudatile {
namespace tblgen {

// Get trait constraint message by name
static std::optional<std::string>
getTraitConstraint(const std::string &traitName) {
  static const std::unordered_map<std::string, std::string> traitMap = {
      {"NoMemoryEffect",
       "The operation is pure and does not perform any memory side effects."},
      {"RecursiveMemoryEffects",
       "The operation only has an effect if and only if it the region's "
       "operation have an effect."},
      {"ConstantLike",
       "The operation has no operands and may be constant folded."},
      {"AlwaysSpeculatableImplTrait",
       "The operation may be speculatively executed without side effects."},
      {"ConditionallySpeculatable",
       "The operation is conditionally speculatable"
       "based on the specific operands and attributes."},
      {"InferTypeOpInterface", "The operation's result type may be inferred "
                               "from its operands and attributes."},
      {"IsolatedFromAbove",
       "The region must not capture SSA values defined above the operation."},
      {"AutomaticAllocationScope", "The operation must define scope when stack "
                                   "allocations are freed automatically."},
      {"SingleBlock", "Each provided region must contain exactly one block."},
      {"NoRegionArguments", "All regions must have zero arguments."},
      {"NoTerminator",
       "The region must not require explicit terminator operations."},
      {"HasOnlyGraphRegion",
       "The operation must contain only dataflow graph regions."},
      {"FunctionOpInterface",
       "The operation must implement function-like behavior interface."},
      {"CallOpInterface", "The operation must implement call interface."},
      {"CallableOpInterface",
       "The operation must implement callable target interface."},
      {"Symbol", "The operation must be a symbol in the global symbol table."},
      {"SymbolTable", "The operation must define a symbol scope."},
      {"OpAsmOpInterface",
       "The operation must provide custom parsing and printing methods."},
      {"AttrSizedOperandSegments", "The operation must encode variadic operand "
                                   "segment sizes in attributes."},
      {"Elementwise", "The operation must apply element-wise to its operands."},
      {"SameOperandsShape", "All operands must have identical shapes."},
      {"Terminator", "The operation must terminate its parent basic block."},
      {"RegionKindInterface",
       "The operation must specify whether regions are SSACFG or Graph kind."},
      {"MemoryEffectOpInterface",
       "The operation must declare its memory effects."},
      {"Pure",
       "The operation has no side effects and may be speculatively executed."},
      {"SameOperandsAndResultType",
       "The operation's operands and results all have the same type."}};

  auto it = traitMap.find(traitName);
  if (it != traitMap.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::string OperationParameter::getDescription() const {
  if (!this->specDesc.empty()) {
    return this->specDesc;
  } else {
    PrintFatalError("The description for " + this->name + " in " +
                    this->opName +
                    " is not available. Please annotate it "
                    "with description using `CudaTileArgMetadata`.");
  }
}

TileIRType OperationParameter::getTypeDescription() const {
  switch (type) {
  case kArgument: {
    if (!typeConstraint) {
      llvm_unreachable("Type constraint is missing");
    }

    auto tcDef = this->typeConstraint->getDef();
    return getType(tcDef);
  }
  case kAttribute: {
    if (!attribute) {
      llvm_unreachable("Attribute is missing");
    }

    return convertAttribute(this->opName, *attribute);
  }
  case kProperty: {
    if (!property) {
      llvm_unreachable("Property is missing");
    }

    PrintFatalError("Property descriptions are not supported yet. Failed while "
                    "processing: " +
                    this->opName + "." + this->name);
  }
  default:
    llvm_unreachable("Invalid parameter type");
  }
}

// Helper function to extract names from a trait's "values" field
static std::vector<std::string>
getTraitValueNames(const llvm::Record &recordDef) {
  std::vector<std::string> names;
  for (const auto &name : recordDef.getValueAsListOfStrings("values")) {
    names.push_back(name.str());
  }
  return names;
}

static std::vector<OperationConstraint>
getOperationConstraints(const mlir::tblgen::Operator &op,
                        OperationSignature &signature) {
  std::vector<OperationConstraint> constraints;

  llvm::errs() << "Processing operation: " << op.getOperationName() << "\n";

  for (auto trait : op.getTraits()) {
    const auto &def = trait.getDef();

    if (def.isSubClassOf("AllTypesMatch")) {
      llvm::errs() << "Processing AllTypesMatch trait\n";
      auto values = def.getValueAsListOfStrings("values");
      std::vector<std::string> names;
      std::string type;
      for (auto val : values) {
        std::string nameStr = val.str();
        names.push_back(nameStr);

        if (auto parameter = signature.getParamOrResult(nameStr)) {
          std::string paramType = parameter->getTypeDescription().toString();

          if (type.empty()) {
            type = paramType;
          } else if (type != paramType) {
            // Skip type constraint check if one of the types is DenseConstant
            if (type == "DenseConstant" ||
                paramType.find("DenseConstant") != std::string::npos) {
              continue;
            }

            llvm::errs() << "Type constraint mismatch detected for name \""
                         << nameStr << "\":\n"
                         << "  Expected based on previous: " << type << "\n"
                         << "  Got: " << paramType << "\n";
            assert(false && "AllTypesMatch: Type constraint mismatch.");
          }
        }
      }
      constraints.push_back(OperationConstraint(AllTypesMatch{names, type}));
    } else if (def.isSubClassOf("AllElementTypeMatch")) {
      llvm::errs() << "Processing AllElementTypeMatch trait\n";
      auto values = def.getValueAsListOfStrings("values");
      std::vector<std::string> names;
      std::string type;
      for (auto val : values) {
        std::string nameStr = val.str();
        names.push_back(nameStr);

        if (auto parameter = signature.getParamOrResult(nameStr)) {
          std::string paramType = parameter->getTypeDescription().toString();

          if (type.empty()) {
            type = paramType;
          } else if (type != paramType) {
            // Skip type constraint check if one of the types is DenseConstant
            if (type == "DenseConstant" ||
                paramType.find("DenseConstant") != std::string::npos) {
              continue;
            }

            llvm::errs() << "Type constraint mismatch detected for name \""
                         << nameStr << "\":\n"
                         << "  Expected based on previous: " << type << "\n"
                         << "  Got: " << paramType << "\n";
            assert(false && "AllTypesMatch: Type constraint mismatch.");
          }
        }
      }
      constraints.push_back(
          OperationConstraint(AllElementTypeMatch{names, type}));
    } else if (def.isSubClassOf("TypesMatchWith")) {
      llvm::errs() << "Processing TypesMatchWith trait\n";
      auto *summaryVal = def.getValue("summary");
      assert(summaryVal &&
             "TypesMatchWith: Trait must have a 'summary' field.");
      auto *stringInit =
          llvm::dyn_cast<llvm::StringInit>(summaryVal->getValue());
      assert(stringInit &&
             "TypesMatchWith: 'summary' field must be a StringInit.");
      constraints.push_back(
          OperationConstraint(TypesMatchWith{stringInit->getValue().str()}));

    } else if (def.isSubClassOf("AnyTypeOf")) {
      llvm::errs() << "Processing AnyTypeOf trait\n";
      constraints.push_back(
          OperationConstraint(AnyTypeOf{getTraitValueNames(def)}));

    } else if (def.isSubClassOf("AllRanksMatch")) {
      llvm::errs() << "Processing AllRanksMatch trait\n";
      constraints.push_back(
          OperationConstraint(AllRanksMatch{getTraitValueNames(def)}));

    } else if (def.getName() == "SameOperandsAndResultElementType") {
      llvm::errs() << "Processing SameOperandsAndResultElementType trait\n";
      std::vector<std::string> names, types;
      for (auto param : signature.parameters) {
        names.push_back(param.name);
        types.push_back(param.getTypeDescription().toString());
      }
      for (auto param : signature.results) {
        names.push_back(param.name);
        types.push_back(param.getTypeDescription().toString());
      }
      if (!types.empty()) {
        assert(
            std::all_of(types.begin() + 1, types.end(),
                        [&](const std::string &t) { return t == types[0]; }) &&
            "All types must be identical");
      }
      constraints.push_back(OperationConstraint(
          SameOperandsAndResultElementType{names, types[0]}));

    } else if (def.getName() == "SameTypeOperands") {
      llvm::errs() << "Processing SameTypeOperands trait\n";
      std::vector<std::string> names;
      std::vector<std::string> types;
      for (auto param : signature.parameters) {
        names.push_back(param.name);
        types.push_back(param.getTypeDescription().toString());
      }
      if (!types.empty()) {
        assert(
            std::all_of(types.begin() + 1, types.end(),
                        [&](const std::string &t) { return t == types[0]; }) &&
            "All types must be identical");
      }
      constraints.push_back(
          OperationConstraint(SameTypeOperands{names, types[0]}));
    } else if (def.getName() == "SameOperandsAndResultShape") {
      llvm::errs() << "Processing SameOperandsAndResultShape trait\n";
      std::vector<std::string> names;
      for (auto param : signature.parameters) {
        names.push_back(param.name);
      }
      for (auto param : signature.results) {
        names.push_back(param.name);
      }
      constraints.push_back(
          OperationConstraint(SameOperandsAndResultShape{names}));
    } else {
      std::string traitName = def.getName().str();
      auto traitConstraint = getTraitConstraint(traitName);
      if (traitConstraint)
        constraints.push_back(
            OperationConstraint(OperationTrait{*traitConstraint}));
    }
  }

  return constraints;
}

OperationSignature::OperationSignature(const mlir::tblgen::Operator &op) {
  auto opName = op.getOperationName();
  this->name = op.getOperationName();
  this->parameters = std::vector<OperationParameter>();
  this->parameterMap = std::unordered_map<std::string, OperationParameter>();

  for (int i = 0; i < op.getNumArgs(); i++) {
    auto arg = op.getArg(i);
    auto argDecorators = op.getArgDecorators(i);

    std::string parameterName;
    std::optional<TypeConstraint> typeConstraint;
    std::optional<Attribute> attribute;
    std::optional<Property> property;

    ParameterType parameterType;

    if (auto *namedTypeConstraint = arg.dyn_cast<NamedTypeConstraint *>()) {
      parameterName = namedTypeConstraint->name;
      typeConstraint = namedTypeConstraint->constraint;
      parameterType = kArgument;
    } else if (auto *namedAttr = arg.dyn_cast<NamedAttribute *>()) {
      parameterName = namedAttr->name;
      attribute = namedAttr->attr;
      parameterType = kAttribute;
    } else if (auto *namedProperty = arg.dyn_cast<NamedProperty *>()) {
      parameterName = namedProperty->name;
      property = namedProperty->prop;
      parameterType = kProperty;
    } else {
      llvm_unreachable("Statically impossible pointer type.");
    }

    std::string sinceVersion;
    std::string specDesc;
    std::optional<std::vector<std::string>> selectedVariants = std::nullopt;

    for (auto argDecorator : argDecorators) {
      if (argDecorator.getDef().isSubClassOf("CudaTileArgMetadata")) {
        sinceVersion = argDecorator.getDef().getValueAsString("sinceVersion");
        specDesc = argDecorator.getDef().getValueAsString("specDesc");
      }
      if (argDecorator.getDef().isSubClassOf("OnlyVariants")) {
        std::cout << "Processing OnlyVariants decorator" << std::endl;
        auto variants =
            argDecorator.getDef().getValueAsListOfStrings("variants");
        std::vector<std::string> foundSelectedVariants;
        for (auto variant : variants) {
          std::cout << "variant: " << variant.str() << std::endl;
          foundSelectedVariants.push_back(variant.str());
        }
        selectedVariants = foundSelectedVariants;
      }
    }

    parameters.push_back({opName, parameterName, sinceVersion, specDesc,
                          typeConstraint, attribute, property, selectedVariants,
                          parameterType});
  }

  for (int i = 0; i < op.getNumResults(); i++) {
    auto result = op.getResult(i);
    auto resultDecorators = op.getResultDecorators(i);

    std::string resultName;
    std::optional<TypeConstraint> typeConstraint;
    std::optional<Attribute> attribute;
    std::optional<Property> property;

    ParameterType parameterType = kArgument;

    resultName = result.name;
    typeConstraint = result.constraint;

    std::string sinceVersion;
    std::string specDesc;
    std::optional<std::vector<std::string>> selectedVariants = std::nullopt;

    for (auto resultDecorator : resultDecorators) {
      if (resultDecorator.getDef().isSubClassOf("CudaTileArgMetadata")) {
        sinceVersion =
            resultDecorator.getDef().getValueAsString("sinceVersion");
        specDesc = resultDecorator.getDef().getValueAsString("specDesc");
      }
      if (resultDecorator.getDef().isSubClassOf("OnlyVariants")) {
        auto variants =
            resultDecorator.getDef().getValueAsListOfStrings("variants");
        std::vector<std::string> foundSelectedVariants;
        for (auto variant : variants) {
          foundSelectedVariants.push_back(variant.str());
        }
        selectedVariants = foundSelectedVariants;
      }
    }

    results.push_back({opName, resultName, sinceVersion, specDesc,
                       typeConstraint, attribute, property, selectedVariants,
                       parameterType});
  }

  for (auto &param : parameters) {
    this->parameterMap.insert({param.name, param});
  }

  for (auto &param : results) {
    this->parameterMap.insert({param.name, param});
  }

  this->constraints = getOperationConstraints(op, *this);
}

CudaTileOp::CudaTileOp(const mlir::tblgen::Operator &op) : op(op) {
  this->signature = OperationSignature(this->op);
}

CudaTileOp::CudaTileOp(const CudaTileOp &other)
    : signature(other.signature), op(other.op) {}

std::string CudaTileOp::getCudaTileSpecGroup() {
  auto def = this->op.getDef();
  auto mdef = def.getValueAsOptionalDef("metadata");

  if (mdef != nullptr) {
    auto cudaTileGroup = mdef->getValueAsOptionalString("cudaTileSpecGroup");
    if (cudaTileGroup) {
      return cudaTileGroup->str();
    }
  }

  return "Miscellanous";
}

std::string CudaTileOp::getCudaTileSpecSubGroup() {
  auto def = this->op.getDef();
  auto mdef = def.getValueAsOptionalDef("metadata");

  if (mdef != nullptr) {
    auto cudaTileSubGroup =
        mdef->getValueAsOptionalString("cudaTileSpecSubGroup");
    if (cudaTileSubGroup) {
      return cudaTileSubGroup->str();
    }
  }

  return "Miscellanous";
}

std::vector<std::string> CudaTileOp::getMLIRExamples() {
  auto def = this->op.getDef();
  auto mdef = def.getValueAsListOfStrings("mlirExamples");
  return std::vector<std::string>(mdef.begin(), mdef.end());
}

static Table getTableFromRecord(const Record *tableDef) {
  std::vector<TableHeader> headers;
  std::vector<TableRow> rows;

  std::string label = tableDef->getValueAsString("label").str();
  std::string description = tableDef->getValueAsString("description").str();
  std::optional<std::string> oDescription = std::nullopt;

  if (!description.empty()) {
    oDescription = description;
  }

  auto headerDefs = tableDef->getValueAsListOfDefs("headers");

  for (auto headerDef : headerDefs) {
    std::string label = headerDef->getValueAsString("label").str();
    std::string contentType = headerDef->getValueAsString("contentType").str();

    ColumnFormatType format;
    if (contentType == "code") {
      format = ColumnFormatType::kCode;
    } else {
      format = ColumnFormatType::kText;
    }

    int width = headerDef->getValueAsInt("width");
    std::optional<int> oWidth =
        width == -1 ? std::nullopt : std::optional<int>(width);

    headers.emplace_back(label, oWidth, format);
  }

  auto rowDefs = tableDef->getValueAsListOfDefs("rows");
  for (auto rowDef : rowDefs) {
    std::vector<std::string> entries;
    for (auto entryDef : rowDef->getValueAsListOfStrings("columns")) {
      entries.push_back(entryDef.str());
    }
    rows.push_back(TableRow{entries});
  }

  // We now have all the headers and the rows, create the table.
  return Table{label, oDescription, headers, rows, {}};
}

std::vector<Table> CudaTileOp::getDescriptionTables() {
  auto def = this->op.getDef();
  auto tableDefs = def.getValueAsListOfDefs("descriptionTables");

  std::vector<Table> tables;
  tables.reserve(tableDefs.size());
  // For each table definition, create a table.
  for (auto tableDef : tableDefs) {
    tables.push_back(getTableFromRecord(tableDef));
  }

  return tables;
}

llvm::StringRef CudaTileOp::getDescription() const {
  return op.getDescription();
}

std::vector<TileIRAttr> CudaTileOp::getAttributes() {
  std::vector<TileIRAttr> attributes;
  std::unordered_set<std::string> processedAttributes;

  for (const auto &it : this->op.getAttributes()) {
    const Record *record = &it.attr.getDef();

    auto attributeName = it.attr.getAttrDefName().str();
    std::cout << "Processing attribute: " << attributeName << std::endl;
    // In the case that we have multiple operands with the same attribute type
    // we only want to generate documentation for the attribute type itself
    // once.
    if (processedAttributes.find(attributeName) != processedAttributes.end()) {
      continue;
    }

    // Strip off the DefaultValuedAttr first.
    if (record->isSubClassOf("DefaultValuedAttr") ||
        record->isSubClassOf("OptionalAttr")) {
      record = record->getValueAsDef("baseAttr");
    }

    // Check for a bare enum value first.
    if (record->isSubClassOf("CudaTileI32EnumAttr") ||
        record->isSubClassOf("CudaTileI64EnumAttr")) {
      auto param = this->signature.getParamOrResult(it.name.str());
      auto enumAttr = TileIREnumAttr::fromTableGen(
          this->op.getOperationName(), *record, param->selectedVariants);
      attributes.emplace_back(enumAttr);
      // Then check to see if it's an cuda tile enum attr.
    } else if (record->isSubClassOf("CudaTileEnumAttr")) {
      std::cout << "Processing CudaTileEnumAttr" << std::endl;
      std::cout << "it.name: " << it.name.str() << std::endl;
      auto param = this->signature.getParamOrResult(it.name.str());
      auto enumInfoRecord = record->getValueAsDef("enum");
      auto enumAttr = TileIREnumAttr::fromTableGen(this->op.getOperationName(),
                                                   *enumInfoRecord,
                                                   param->selectedVariants);
      attributes.emplace_back(enumAttr);
    } else if (record->isSubClassOf("CudaTileAttrDef")) {
      auto attrDef =
          TileIRAttrDef::fromTableGen(this->op.getOperationName(), *record);
      attributes.emplace_back(attrDef);
    } else if (record->isSubClassOf("AttrInterface")) {
      auto name = record->getName();
      // Remove the "CudaTile_" prefix from the attribute name if present.
      static const std::string prefix = "CudaTile_";
      if (name.starts_with(prefix)) {
        name = name.substr(prefix.size());
      }
      static const std::string suffix = "AttrInterface";
      if (name.ends_with(suffix)) {
        name = name.substr(0, name.size() - suffix.size());
      }

      auto attrInterface =
          TileIRAttrInterface{this->op.getOperationName(), name.str(), "", ""};
      attributes.emplace_back(attrInterface);
    }

    processedAttributes.insert(attributeName);
  }
  return attributes;
}

} // namespace tblgen
} // namespace cudatile
