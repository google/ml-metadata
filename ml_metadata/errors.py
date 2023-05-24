# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exception types for MLMD errors."""
from absl import logging

# The error code values are aligned with absl status errors.
# TODO(b/143236826) Drop this once absl status is available in python.
OK = 0
CANCELLED = 1
UNKNOWN = 2
INVALID_ARGUMENT = 3
DEADLINE_EXCEEDED = 4
NOT_FOUND = 5
ALREADY_EXISTS = 6
PERMISSION_DENIED = 7
UNAUTHENTICATED = 16
RESOURCE_EXHAUSTED = 8
FAILED_PRECONDITION = 9
ABORTED = 10
OUT_OF_RANGE = 11
UNIMPLEMENTED = 12
INTERNAL = 13
UNAVAILABLE = 14
DATA_LOSS = 15


class StatusError(Exception):
  """A general error class that cast maps Status to typed errors."""

  def __init__(self, message, error_code):
    """Creates a `StatusError`."""
    super(StatusError, self).__init__(message)
    self.message = message
    self.error_code = error_code


class CancelledError(StatusError):
  """Raised when an operation or step is cancelled."""

  def __init__(self, message):
    """Creates a `CancelledError`."""
    super(CancelledError, self).__init__(message, CANCELLED)


class UnknownError(StatusError):
  """Raised when an operation failed reason is unknown."""

  def __init__(self, message, error_code=UNKNOWN):
    """Creates an `UnknownError`."""
    super(UnknownError, self).__init__(message, error_code)


class InvalidArgumentError(StatusError):
  """Raised when an operation receives an invalid argument."""

  def __init__(self, message):
    """Creates an `InvalidArgumentError`."""
    super(InvalidArgumentError, self).__init__(message, INVALID_ARGUMENT)


class DeadlineExceededError(StatusError):
  """Raised when a deadline expires before an operation could complete."""

  def __init__(self, message):
    """Creates a `DeadlineExceededError`."""
    super(DeadlineExceededError, self).__init__(message, DEADLINE_EXCEEDED)


class NotFoundError(StatusError):
  """Raised when a requested entity was not found."""

  def __init__(self, message):
    """Creates a `NotFoundError`."""
    super(NotFoundError, self).__init__(message, NOT_FOUND)


class AlreadyExistsError(StatusError):
  """Raised when an entity that we attempted to create already exists."""

  def __init__(self, message):
    """Creates an `AlreadyExistsError`."""
    super(AlreadyExistsError, self).__init__(message, ALREADY_EXISTS)


class PermissionDeniedError(StatusError):
  """Raised when the caller does not have permission to run an operation."""

  def __init__(self, message):
    """Creates a `PermissionDeniedError`."""
    super(PermissionDeniedError, self).__init__(message, PERMISSION_DENIED)


class UnauthenticatedError(StatusError):
  """The request does not have valid authentication credentials."""

  def __init__(self, message):
    """Creates an `UnauthenticatedError`."""
    super(UnauthenticatedError, self).__init__(message, UNAUTHENTICATED)


class ResourceExhaustedError(StatusError):
  """Some resource has been exhausted."""

  def __init__(self, message):
    """Creates a `ResourceExhaustedError`."""
    super(ResourceExhaustedError, self).__init__(message, RESOURCE_EXHAUSTED)


class FailedPreconditionError(StatusError):
  """Raised when the system is not in a state to execute an operation."""

  def __init__(self, message):
    """Creates a `FailedPreconditionError`."""
    super(FailedPreconditionError, self).__init__(message, FAILED_PRECONDITION)


class AbortedError(StatusError):
  """The operation was aborted, typically due to a concurrent action."""

  def __init__(self, message):
    """Creates an `AbortedError`."""
    super(AbortedError, self).__init__(message, ABORTED)


class OutOfRangeError(StatusError):
  """Raised when an operation iterates past the valid input range."""

  def __init__(self, message):
    """Creates an `OutOfRangeError`."""
    super(OutOfRangeError, self).__init__(message, OUT_OF_RANGE)


class UnimplementedError(StatusError):
  """Raised when an operation has not been implemented."""

  def __init__(self, message):
    """Creates an `UnimplementedError`."""
    super(UnimplementedError, self).__init__(message, UNIMPLEMENTED)


class InternalError(StatusError):
  """Raised when the system experiences an internal error."""

  def __init__(self, message):
    """Creates an `InternalError`."""
    super(InternalError, self).__init__(message, INTERNAL)


class UnavailableError(StatusError):
  """Raised when the runtime is currently unavailable."""

  def __init__(self, message):
    """Creates an `UnavailableError`."""
    super(UnavailableError, self).__init__(message, UNAVAILABLE)


class DataLossError(StatusError):
  """Raised when unrecoverable data loss or corruption is encountered."""

  def __init__(self, message):
    """Creates a `DataLossError`."""
    super(DataLossError, self).__init__(message, DATA_LOSS)


_CODE_TO_EXCEPTION_CLASS = {
    CANCELLED: CancelledError,
    UNKNOWN: UnknownError,
    INVALID_ARGUMENT: InvalidArgumentError,
    DEADLINE_EXCEEDED: DeadlineExceededError,
    NOT_FOUND: NotFoundError,
    ALREADY_EXISTS: AlreadyExistsError,
    PERMISSION_DENIED: PermissionDeniedError,
    UNAUTHENTICATED: UnauthenticatedError,
    RESOURCE_EXHAUSTED: ResourceExhaustedError,
    FAILED_PRECONDITION: FailedPreconditionError,
    ABORTED: AbortedError,
    OUT_OF_RANGE: OutOfRangeError,
    UNIMPLEMENTED: UnimplementedError,
    INTERNAL: InternalError,
    UNAVAILABLE: UnavailableError,
    DATA_LOSS: DataLossError,
}


def exception_type_from_error_code(error_code):
  """Returns error class w.r.t. the error_code."""
  return _CODE_TO_EXCEPTION_CLASS[error_code]


def make_exception(message: str, error_code: int):
  """Makes an exception with the MLMD error code.

  Args:
    message: Error message.
    error_code: MLMD error code.

  Returns:
    An exception.
  """

  try:
    exc_type = exception_type_from_error_code(error_code)
    # log internal backend engine errors only.
    if error_code == INTERNAL:
      logging.log(
          logging.WARNING, 'mlmd client %s: %s', exc_type.__name__, message
      )
    return exc_type(message)
  except KeyError:
    return UnknownError(message)
