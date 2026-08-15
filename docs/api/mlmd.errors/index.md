# mlmd.errors

Exception types for MLMD errors.

## Classes

[`class AbortedError`][ml_metadata.errors.AbortedError]: The operation was aborted, typically due to a concurrent action.

[`class AlreadyExistsError`][ml_metadata.errors.AlreadyExistsError]: Raised when an entity that we attempted to create already exists.

[`class CancelledError`][ml_metadata.errors.CancelledError]: Raised when an operation or step is cancelled.

[`class DataLossError`][ml_metadata.errors.DataLossError]: Raised when unrecoverable data loss or corruption is encountered.

[`class DeadlineExceededError`][ml_metadata.errors.DeadlineExceededError]: Raised when a deadline expires before an operation could complete.

[`class FailedPreconditionError`][ml_metadata.errors.FailedPreconditionError]: Raised when the system is not in a state to execute an operation.

[`class InternalError`][ml_metadata.errors.InternalError]: Raised when the system experiences an internal error.

[`class InvalidArgumentError`][ml_metadata.errors.InvalidArgumentError]: Raised when an operation receives an invalid argument.

[`class NotFoundError`][ml_metadata.errors.NotFoundError]: Raised when a requested entity was not found.

[`class OutOfRangeError`][ml_metadata.errors.OutOfRangeError]: Raised when an operation iterates past the valid input range.

[`class PermissionDeniedError`][ml_metadata.errors.PermissionDeniedError]: Raised when the caller does not have permission to run an operation.

[`class ResourceExhaustedError`][ml_metadata.errors.ResourceExhaustedError]: Some resource has been exhausted.

[`class StatusError`][ml_metadata.errors.StatusError]: A general error class that cast maps Status to typed errors.

[`class UnauthenticatedError`][ml_metadata.errors.UnauthenticatedError]: The request does not have valid authentication credentials.

[`class UnavailableError`][ml_metadata.errors.UnavailableError]: Raised when the runtime is currently unavailable.

[`class UnimplementedError`][ml_metadata.errors.UnimplementedError]: Raised when an operation has not been implemented.

[`class UnknownError`][ml_metadata.errors.UnknownError]: Raised when an operation failed reason is unknown.

## Functions

[`exception_type_from_error_code(...)`][ml_metadata.errors.exception_type_from_error_code]: Returns error class w.r.t. the error_code.

[`make_exception(...)`][ml_metadata.errors.make_exception]: Makes an exception with the MLMD error code.
