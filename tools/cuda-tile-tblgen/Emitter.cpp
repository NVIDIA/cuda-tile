//===- Emitter.cpp - CUDA Tile dialect spec generator helpers ---*- C++ -*-===//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Emitter.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"

#include <fstream>

using namespace llvm;
using namespace mlir;
using namespace cudatile::tblgen;

namespace cudatile {
namespace tblgen {

raw_ostream &operator<<(raw_ostream &os, const Code &code) {
  os << std::string(code);
  return os;
}

raw_ostream &operator<<(raw_ostream &os, const TileIRTy &type) {
  os << std::string(type);
  return os;
}

raw_ostream& operator<<(raw_ostream& os, const Header& header) {
    char underline;
    switch (header.level) {
        case 1: underline = '#'; break;
        case 2: underline = '*'; break;
        case 3: underline = '='; break;
        case 4: underline = '-'; break;
        case 5: underline = '^'; break;
        case 6: underline = '"'; break;
        default: underline = '"'; break;
    }

    // Generate a string where the header.title is on one line and the underline is the same length.
    std::string underline_str(header.title.size(), underline);
    os << header.title << "\n" << underline_str << "\n";
    return os;
}

raw_ostream &operator<<(raw_ostream &os, const Table &table) {
  os << table.description << "\n\n";
  for (auto &anchor : table.anchors) {
    os << ".. _" << anchor << ":\n\n";
  }
  os << ".. list-table:: " << table.title << "\n";

  std::string widths;
  for (auto &header : table.headers) {
    if (header.width) {
      widths += std::to_string(header.width.value()) + " ";
    }
  };

  if (!widths.empty()) {
    os << "   :widths: " << widths << "\n";
  }

  // For now only row of headers.
  os << "   :header-rows: 1\n\n";

  for (size_t i = 0; i < table.headers.size(); ++i) {
    if (i == 0) {
      os << "   * - ";
    } else {
      os << "     - ";
    }
    os << table.headers[i].title << "\n";
  }

  for (auto &row : table.rows) {
    for (size_t i = 0; i < row.columns.size(); ++i) {
      if (i == 0) {
        os << "   * - ";
      } else {
        os << "     - ";
      }

      if (table.headers[i].format == ColumnFormatType::kCode) {
        os << ":code:`" << row.columns[i] << "`";
      } else {
        os << row.columns[i];
      }

      os << "\n";
    }
  }

  return os;
}

raw_ostream &operator<<(raw_ostream &os, const Badge &badge) {
  switch (badge.type) {
  case BadgeType::kPrimary:
    os << ":bdg-ref-primary`" << badge.text << "`";
    break;
  case BadgeType::kPrimaryLine:
    os << ":bdg-ref-primary-line:`" << badge.text << "`";
    break;
  case BadgeType::kSuccess:
    os << ":bdg-success:`" << badge.text << "`";
    break;
  case BadgeType::kSuccessLine:
    os << ":bdg-success-line:`" << badge.text << "`";
    break;
  case BadgeType::kInfo:
    os << ":bdg-info:`" << badge.text << "`";
    break;
  case BadgeType::kInfoLine:
    os << ":bdg-info-line:`" << badge.text << "`";
    break;
  case BadgeType::kWarning:
    os << ":bdg-warning:`" << badge.text << "`";
    break;
  case BadgeType::kWarningLine:
    os << ":bdg-warning-line:`" << badge.text << "`";
    break;
  case BadgeType::kDanger:
    os << ":bdg-danger:`" << badge.text << "`";
    break;
  case BadgeType::kDangerLine:
    os << ":bdg-danger-line:`" << badge.text << "`";
    break;
  }

  return os;
}

inline std::string
examplesAppendixFile(const std::optional<std::string> &examplesDirectory) {
  return examplesDirectory.value() + "/examples_appendix.rst";
}

SpecEmitter::SpecEmitter(raw_indented_ostream &os,
                         const std::optional<std::string> &examplesDirectory)
    : os(os), examplesDirectory(examplesDirectory) {
  this->appendixFile = std::ofstream();
  this->appendixFile.open(examplesAppendixFile(examplesDirectory));
}

void SpecEmitter::emitLiteralInclude(
    const std::string &fileName, const std::string &anchor,
    const std::vector<std::tuple<int, int>> &lineRanges,
    const std::string &language, const std::optional<int> dedent) {
  this->os << ".. literalinclude:: " << fileName << "\n";
  if (!lineRanges.empty()) {
    this->os << indent << ":lines: ";
    size_t i = 0;
    for (const auto &lines : lineRanges) {
      this->os << std::get<0>(lines) << "-" << std::get<1>(lines);
      i++;
      if (i != lineRanges.size()) {
        this->os << ",";
      }
    }
    this->os << "\n";
  }
  this->os << indent << ":language: " << language << "\n";
  if (dedent) {
    this->os << indent << ":dedent: " << dedent.value() << "\n";
  }
  this->os << "\n";
}

void SpecEmitter::writeExampleToDiskAndAppendToAppendix(
    const std::string &exampleName, const std::string &exampleAnchor,
    const std::string &fileName, const std::string &example) {
  // If the example directory is not set, do nothing.
  if (!this->examplesDirectory) {
    return;
  }

  llvm::SmallVector<char, 128> filePath;
  // The path to write the example file to in the build directory.
  llvm::sys::path::append(filePath, this->examplesDirectory.value(), fileName);

  // The relative path to the example file in the spec.
  std::string relativePath = "/_spec_gen/examples/" + fileName;

  // Add an anchor to the example
  this->appendixFile << ".. _" << exampleAnchor << ":\n\n";

  // Add example name as header and example content
  auto exampleNameWithIgnore = ":spelling:ignore:`" + exampleName + "`";
  this->appendixFile << "\n" << exampleNameWithIgnore << "\n";
  this->appendixFile << std::string(exampleNameWithIgnore.length(), '~')
                     << "\n\n";
  this->appendixFile << ".. literalinclude:: " << relativePath << "\n";
  this->appendixFile << indent << ":language: "
                     << "mlir"
                     << "\n\n";
  // Indent example content
  // Create directories if they don't exist
  std::error_code ec =
      llvm::sys::fs::create_directories(this->examplesDirectory.value());

  if (ec) {
    llvm::errs() << "Error creating directory "
                 << this->examplesDirectory.value() << ": " << ec.message()
                 << "\n";
    return;
  }

  // Open file for writing
  std::error_code writeEC;
  llvm::raw_fd_ostream outFile(std::string(filePath.begin(), filePath.end()),
                               writeEC);
  if (writeEC) {
    llvm::errs() << "Error opening file " << filePath << ": "
                 << writeEC.message() << "\n";
    return;
  }

  // Write content to file
  outFile << example;
  outFile.close();
}

void SpecEmitter::emitExample(const std::string &exampleName,
                              const FormattedExample &formattedExample) {

  auto exampleFileName = "example_" + exampleName + ".mlir";
  auto exampleAnchor = "example_" + exampleName;
  auto exampleFilePath = "/_spec_gen/examples/" + exampleFileName;
  this->writeExampleToDiskAndAppendToAppendix(
      exampleName, exampleAnchor, exampleFileName, formattedExample.content);

  this->emitLiteralInclude(exampleFilePath, exampleAnchor,
                           formattedExample.lineRanges, "mlir",
                           formattedExample.dedent);

  // Investigate whether we can attach this as caption text to the example.
  this->os << "See :ref:`" << exampleAnchor
           << "` for the full example listing.\n\n";
}

} // namespace tblgen
} // namespace cudatile
