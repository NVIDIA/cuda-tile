//===- Remark.h -------------------------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_CUDATILE_IR_REMARK_H
#define CUDATILE_DIALECT_CUDATILE_IR_REMARK_H

//===----------------------------------------------------------------------===//
//
// TILE IR REMARK SYSTEM - DEVELOPER GUIDE
// ========================================
//
// HOW TO ADD A NEW REMARK
//
// Step 1: Edit Remarks.td
// (a) Choose a category (or create one):
//     - CategoryTensorCore : Tensor Core / MMA operations
//     - CategoryMemory     : Memory operations (TMA, loads, stores)
//
//     To add a new category:
//       def CategoryMyNew : TileIRCategory<"MyNew", 3, "My-Description">;
//     Then add it to the CategoryID enum at the bottom of Remarks.td.
//
// (b) Choose a diagnostic kind:
//     - DK_Pass     : Optimization succeeded
//     - DK_Miss     : Optimization opportunity missed
//     - DK_Fail     : Optimization attempted but failed
//     - DK_Analysis : Informational analysis (no pass/fail)
//
// (b2) Special always-on diagnostics (not subject to remark filtering):
//     - TileIRError<>   + reportError()
//     - TileIRWarning<> + reportWarning()
//
//     These do not take a category and always route through the MLIR remark
//     engine via RemarkEngine::report(), bypassing the per-kind filters so
//     the diagnostic is always visible whenever a remark engine is attached
//     to the context. Each one carries a `RemarkId` argument such as
//     `E0000` / `W0000` (see the ID note below).
//
// (c) Define the remark:
//
//     def RemarkMyOptimization
//       : TileIRRemark<"RemarkMyOptimization",  // Name
//                      DK_Pass,                 // Kind
//                      CategoryMemory,          // Category
//                      10,                      // Unique ID
//                      "Optimized successfully with {0}">;  // Message
//
//     Message templates support {0}, {1}, etc. for runtime formatting.
//
// (d) Register in RemarkID enum (bottom of Remarks.td):
//
//     def RemarkID : I32EnumAttr<"RemarkID", "...",
//         [RemarkTensorCoreMMA,
//          RemarkMemoryLoadInstructionSelected,
//          RemarkMyOptimization,    // <-- Add here
//          ...]> { ... }
//
//
// Step 2: Emit the Remark in C++ Code
// Include this header and call reportRemark():
//   auto rmk = remark::reportRemark(
//       op, // MLIR Operation*
//       remark::RemarkID::RemarkMyOptimization_Succeeded);
//   rmk << "details for {0}: " << anAttribute << " and some other information";
//   rmk.addReason(<optional operation>) << "why it happened"; // optional
//   rmk.addSuggestion(<optional location>) << "what the user can do";
// Or, if you have everything up front and it's already a string:
//   remark::reportRemark(
//      op, // MLIR Operation*
//      remark::RemarkID::RemarkMyOptimization_Succeeded,
//      "extra details",
//      "why it happened", // optional
//      "what the user can do" // optional
//     )
//
// Note: The C++ enum name is "<RemarkName>_<KindSuffix>":
//   - DK_Pass     -> _Succeeded
//   - DK_Miss     -> _Missed
//   - DK_Fail     -> _Failed
//   - DK_Analysis -> _Analysis
//
// Error / Warning entries live in their own enums (ErrorID / WarningID)
// so call sites state intent directly. The enum name has no kind suffix,
// because each entry is a single symbol in its category:
//
//   remark::reportError(op, remark::ErrorID::GenericError, "what happened");
//   remark::reportWarning(op, remark::WarningID::GenericWarning, "what");
//
// reportError is a diagnostic emitter — it does NOT end compilation.
// Callers that need the pass / op to fail must propagate failure
// themselves, e.g.:
//
//   remark::reportError(op, remark::ErrorID::GenericError, "...");
//   signalPassFailure();            // inside a Pass
//   return failure();               // or inside an op verifier / helper
//
// User-facing IDs come straight from the TableGen `Num` field: `Num=N`
// becomes the `RemarkId` argument `E####` / `W####` where `####` is `N`
// zero-padded to four digits (so `Num=0` is `E0000`, `Num=7` is `E0007`):
//
//   TableGen:  def GenericError
//                : TileIRError<"GenericError", 0,
//                              "Internal compiler error: {0}">;
//   RemarkId:  E0000
//
//   TableGen:  def GenericWarning : TileIRWarning<"GenericWarning", 0, "...">;
//   RemarkId:  W0000
//
// When adding a new Error/Warning, pick the next unused `Num` and append
// it to the ErrorID / WarningID enum list in Errors.td / Warnings.td.
// `Num` is the public identifier — do not reuse or renumber existing
// entries.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h" // For SymbolOpInterface
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include "cuda_tile/Dialect/CudaTile/IR/TileIRRemarks.h.inc"

//------------------------------------------------------------------------------
//  Diagnostic emitters
//------------------------------------------------------------------------------

namespace mlir::cuda_tile::remark {

/// This class provides a way to build up a string for an InFlightRemark. This
/// is somewhat specialized for two reasons; first, we have this AddNoteCallback
/// that's used for when we want to stream symbol ops into the string - that's
/// used to avoid having the full IR of the op stored in the string. The second
/// reason is that we're building up a message that's owned by the
/// InFlightRemark, so we're essentially just wrapping helper functions around
/// llvm::raw_string_ostream.
class InFlightString {
  /// Callback that we can use to have InFlightString call back into the
  /// InFlightRemark addNote functionality opaquely.
  using AddNoteCallback =
      llvm::function_ref<void(Location, Attribute, Attribute)>;

public:
  InFlightString(AddNoteCallback addNote, std::string &message)
      : addNote(addNote), message(message) {}

  /// Anything that defines llvm::raw_ostream &operator<< can work here, it just
  /// gets streamed directly into the string.
  template <typename T>
  InFlightString &operator<<(T t) {
    llvm::raw_string_ostream{message} << t;
    return *this;
  }

  /// Symbols are handled a little differently - we add the symbol name as a
  /// reference to the string and attach a note with the symbol's definition
  /// location. This is convenient because most symbols are large, and it
  /// doesn't make sense to print the whole symbol inline.
  InFlightString &operator<<(SymbolOpInterface sym) {
    *this << SymbolRefAttr::get(sym.getNameAttr());
    addNote(sym.getLoc(), sym.getNameAttr(),
            StringAttr::get(sym.getContext(), "definition"));
    return *this;
  }

private:
  AddNoteCallback addNote;
  std::string &message;
};

using RemarkID = cuda_tile::remark::RemarkID;
using ErrorID = cuda_tile::remark::ErrorID;
using WarningID = cuda_tile::remark::WarningID;

/// This class provides an in-flight remark that's still being built. The remark
/// takes a configurable finalizer that is called when the remark is destroyed,
/// and otherwise is a centralized place to do things like add notes,
/// suggestions, and reasons. The op that caused the remark may not be the
/// reason why the remark was emitted, so we allow for reasons to be attached to
/// the op they're causally attached-to. Similarly, notes and suggestions may be
/// attached to a location slightly different from the remark location, and so
/// the user can provide an optional location for suggestions and notes if
/// desired to indicate where the note or suggestion would be best understood as
/// 'attached'.
class InFlightRemark {
public:
  /// Provide a finalizer. The finalizer may fail, in which case we fall back to
  /// a default finalizer. Same for a missing finalizer - we'll fall back to a
  /// default.
  using Finalizer =
      llvm::function_ref<LogicalResult(Operation *, InFlightRemark &)>;

  /// InFlightRemark requires a finalizer, an operation the remark is 'about',
  /// and an ID/name.
  InFlightRemark(Finalizer finalizer, Operation *op, RemarkID id)
      : finalizer(finalizer), op(op), name(id) {}
  ~InFlightRemark();

  //===--------------------------------------------------------------------===//
  // Stream Operators
  //===--------------------------------------------------------------------===//

  /// In general, anything streamed into a remark will be appended to the
  /// message. This is possible to chain like any other ostream call.
  template <typename T>
  InFlightRemark &operator<<(T t) {
    llvm::raw_string_ostream(message) << t;
    return *this;
  }

  /// For symbols specifically, if we stream those in we always only attach the
  /// name as a symbol ref and then add a note for the definition.
  InFlightRemark &operator<<(SymbolOpInterface sym) {
    *this << SymbolRefAttr::get(sym.getNameAttr());
    // The symbol name comes first because that'd be the unique thing that goes
    // into a dict-like object.
    addNote(sym.getLoc(), sym.getNameAttr(),
            StringAttr::get(sym.getContext(), "definition"));
    return *this;
  }

  //===--------------------------------------------------------------------===//
  // addSuggestion/addNote/addReason
  //===--------------------------------------------------------------------===//

  /// Suggestions can be at the remark's location or at a different location -
  /// whichever makes the most sense for the specific use case.
  InFlightString addSuggestion(std::optional<Location> loc = std::nullopt);

  /// Notes can also be attached at the remark's location or at a different
  /// location. Notes are fundamentally key/value pairs of Attribute:Attribute.
  /// Users are free to choose any attribute they want, however.
  void addNote(std::optional<Location> loc, Attribute key, Attribute value) {
    notes.push_back({loc, key, value});
  }
  void addNote(Attribute key, Attribute value) {
    notes.push_back({std::nullopt, key, value});
  }
  void addNote(std::optional<Location> loc, llvm::StringRef key,
               Attribute value) {
    notes.push_back({loc, StringAttr::get(value.getContext(), key), value});
  }
  void addNote(llvm::StringRef key, Attribute value) {
    notes.push_back(
        {std::nullopt, StringAttr::get(value.getContext(), key), value});
  }

  /// Add a new reason. Because the reason is attached to the operation that
  /// it's about, the user must pass in the operation to attach the reason *to*.
  /// If no operation is provided, we default to the operation the
  /// InFlightRemark is about.
  InFlightString addReason(Operation *reasonOp);

  //===--------------------------------------------------------------------===//
  // Getters
  //===--------------------------------------------------------------------===//

  /// Get the ID for this remark.
  RemarkID getName() { return name; }

  /// Get the rendered message for this remark.
  StringRef getMessage() { return message; }

  /// Notes are what we use to store key/value structured data. Notes are
  /// usually string:any pairs, but can in theory contain anything in the key
  /// field. This is especially useful for generating machine-readable notes, as
  /// we could simply have the keys be integers, or enumerations, or really
  /// anything we want.
  struct Note {
    std::optional<Location> loc;
    Attribute key;
    Attribute value;
  };
  ArrayRef<Note> getNotes() const { return notes; }

  /// Suggestions are how we provide the user with actionable feedback.
  /// Suggestions can be attached to any location or the location of the remark
  /// itself.
  struct Suggestion {
    std::optional<Location> loc;
    std::string message;
  };
  ArrayRef<Suggestion> getSuggestions() const { return suggestions; }

  /// Reasons are another way to provide the user with actionable feedback.
  /// Reasons are always attached to the operation that 'causes' the remark,
  /// which can be the operation the remark itself is attached to.
  struct Reason {
    Operation *op;
    std::string message;
  };
  ArrayRef<Reason> getReasons() const { return reasons; }

private:
  /// Called in the destructor, used to allow for configurability.
  Finalizer finalizer;

  /// This is the op that the remark will be attached to when we're done
  /// constructing it.
  Operation *op;

  /// The name of the remark.
  RemarkID name;

  /// This is the message that we can build up through the streaming interface.
  std::string message;

  SmallVector<Note, 2> notes;
  SmallVector<Suggestion, 2> suggestions;
  SmallVector<Reason, 1> reasons;
};

/// Emit a remark with a user-specified finalizer. The user-specified finalizer
/// may fail, in which case we will fall back to the default finalizer.
InFlightRemark reportRemark(Operation *op, RemarkID id,
                            InFlightRemark::Finalizer finalizer = {});

/// Atomic single-function-call version of reportRemark above. This simply
/// pre-populates the remark with the message, one reason, and one suggestion
/// since we know them at function call time.
InFlightRemark reportRemark(Operation *op, RemarkID id, StringRef message,
                            StringRef reason = {}, StringRef suggestion = {},
                            InFlightRemark::Finalizer finalizer = {});

/// Report an error. Always emitted (bypasses remark filtering).
///
/// Does NOT end compilation. reportError is a diagnostic emitter; the
/// caller is responsible for propagating failure — e.g. by calling
/// signalPassFailure() inside a pass, or returning failure() from the
/// operation that detected the error.
void reportError(Operation *op, ErrorID id, StringRef extraMsg = {},
                 StringRef reason = {}, StringRef suggestion = {});

/// Report a warning. Always emitted (bypasses remark filtering).
void reportWarning(Operation *op, WarningID id, StringRef extraMsg = {},
                   StringRef reason = {}, StringRef suggestion = {});

} // namespace mlir::cuda_tile::remark

#endif // CUDATILE_DIALECT_CUDATILE_IR_REMARK_H
