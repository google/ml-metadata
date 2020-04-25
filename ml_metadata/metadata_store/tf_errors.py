# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Exception types for TensorFlow errors."""

import inspect


class OpError(Exception):
  """A generic error that is raised when TensorFlow execution fails.

  Whenever possible, the session will raise a more specific subclass
  of `OpError` from the `tf.errors` module.
  """

  def __init__(self, node_def, op, message, error_code):
    """Creates a new `OpError` indicating that a particular op failed.

    Args:
      node_def: The `node_def_pb2.NodeDef` proto representing the op that
        failed, if known; otherwise None.
      op: The `ops.Operation` that failed, if known; otherwise None.
      message: The message string describing the failure.
      error_code: The `error_codes_pb2.Code` describing the error.
    """
    super(OpError, self).__init__()
    self._node_def = node_def
    self._op = op
    self._message = message
    self._error_code = error_code

  def __reduce__(self):
    # Allow the subclasses to accept less arguments in their __init__.
    init_argspec = inspect.getargspec(self.__class__.__init__)
    args = tuple(getattr(self, arg) for arg in init_argspec.args[1:])
    return self.__class__, args

  @property
  def message(self):
    """The error message that describes the error."""
    return self._message

  @property
  def op(self):
    """The operation that failed, if known.

    *N.B.* If the failed op was synthesized at runtime, e.g. a `Send`
    or `Recv` op, there will be no corresponding
    `tf.Operation`
    object.  In that case, this will return `None`, and you should
    instead use the `tf.errors.OpError.node_def` to
    discover information about the op.

    Returns:
      The `Operation` that failed, or None.
    """
    return self._op

  @property
  def error_code(self):
    """The integer error code that describes the error."""
    return self._error_code

  @property
  def node_def(self):
    """The `NodeDef` proto representing the op that failed."""
    return self._node_def

  def __str__(self):
    return self.message

# The codes are based on https://github.com/tensorflow/tensorflow/blob/b1e813e2ec9634ec0e6562b836e372e393f3de43/tensorflow/core/protobuf/error_codes.proto#L26
OK = 0  # error_codes_pb2.OK
CANCELLED = 1  # error_codes_pb2.CANCELLED
UNKNOWN = 2  # error_codes_pb2.UNKNOWN
INVALID_ARGUMENT = 3  # error_codes_pb2.INVALID_ARGUMENT
DEADLINE_EXCEEDED = 4  # error_codes_pb2.DEADLINE_EXCEEDED
NOT_FOUND = 5  # error_codes_pb2.NOT_FOUND
ALREADY_EXISTS = 6  # error_codes_pb2.ALREADY_EXISTS
PERMISSION_DENIED = 7  # error_codes_pb2.PERMISSION_DENIED
UNAUTHENTICATED = 16  # error_codes_pb2.UNAUTHENTICATED
RESOURCE_EXHAUSTED = 8  # error_codes_pb2.RESOURCE_EXHAUSTED
FAILED_PRECONDITION = 9  # error_codes_pb2.FAILED_PRECONDITION
ABORTED = 10  # error_codes_pb2.ABORTED
OUT_OF_RANGE = 11  # error_codes_pb2.OUT_OF_RANGE
UNIMPLEMENTED = 12  # error_codes_pb2.UNIMPLEMENTED
INTERNAL = 13  # error_codes_pb2.INTERNAL
UNAVAILABLE = 14  # error_codes_pb2.UNAVAILABLE
DATA_LOSS = 15  # error_codes_pb2.DATA_LOSS


# pylint: disable=line-too-long
class CancelledError(OpError):
  """Raised when an operation or step is cancelled.

  For example, a long-running operation (e.g.
  `tf.QueueBase.enqueue` may be
  cancelled by running another operation (e.g.
  `tf.QueueBase.close`,
  or by `tf.Session.close`.
  A step that is running such a long-running operation will fail by raising
  `CancelledError`.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates a `CancelledError`."""
    super(CancelledError, self).__init__(node_def, op, message, CANCELLED)
# pylint: enable=line-too-long


class UnknownError(OpError):
  """Unknown error.

  An example of where this error may be returned is if a Status value
  received from another address space belongs to an error-space that
  is not known to this address space. Also, errors raised by APIs that
  do not return enough error information may be converted to this
  error.

  @@__init__
  """

  def __init__(self, node_def, op, message, error_code=UNKNOWN):
    """Creates an `UnknownError`."""
    super(UnknownError, self).__init__(node_def, op, message, error_code)


class InvalidArgumentError(OpError):
  """Raised when an operation receives an invalid argument.

  This may occur, for example, if an operation receives an input
  tensor that has an invalid value or shape. For example, the
  `tf.matmul` op will raise this
  error if it receives an input that is not a matrix, and the
  `tf.reshape` op will raise
  this error if the new shape does not match the number of elements in the input
  tensor.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates an `InvalidArgumentError`."""
    super(InvalidArgumentError, self).__init__(node_def, op, message,
                                               INVALID_ARGUMENT)


class DeadlineExceededError(OpError):
  """Raised when a deadline expires before an operation could complete.

  This exception is not currently used.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates a `DeadlineExceededError`."""
    super(DeadlineExceededError, self).__init__(node_def, op, message,
                                                DEADLINE_EXCEEDED)


class NotFoundError(OpError):
  """Raised when a requested entity (e.g., a file or directory) was not found.

  For example, running the
  `tf.WholeFileReader.read`
  operation could raise `NotFoundError` if it receives the name of a file that
  does not exist.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates a `NotFoundError`."""
    super(NotFoundError, self).__init__(node_def, op, message, NOT_FOUND)


class AlreadyExistsError(OpError):
  """Raised when an entity that we attempted to create already exists.

  For example, running an operation that saves a file
  (e.g. `tf.train.Saver.save`)
  could potentially raise this exception if an explicit filename for an
  existing file was passed.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates an `AlreadyExistsError`."""
    super(AlreadyExistsError, self).__init__(node_def, op, message,
                                             ALREADY_EXISTS)


class PermissionDeniedError(OpError):
  """Raised when the caller does not have permission to run an operation.

  For example, running the
  `tf.WholeFileReader.read`
  operation could raise `PermissionDeniedError` if it receives the name of a
  file for which the user does not have the read file permission.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates a `PermissionDeniedError`."""
    super(PermissionDeniedError, self).__init__(node_def, op, message,
                                                PERMISSION_DENIED)


class UnauthenticatedError(OpError):
  """The request does not have valid authentication credentials.

  This exception is not currently used.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates an `UnauthenticatedError`."""
    super(UnauthenticatedError, self).__init__(node_def, op, message,
                                               UNAUTHENTICATED)


class ResourceExhaustedError(OpError):
  """Some resource has been exhausted.

  For example, this error might be raised if a per-user quota is
  exhausted, or perhaps the entire file system is out of space.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates a `ResourceExhaustedError`."""
    super(ResourceExhaustedError, self).__init__(node_def, op, message,
                                                 RESOURCE_EXHAUSTED)


class FailedPreconditionError(OpError):
  """Operation was rejected because the system is not in a state to execute it.

  This exception is most commonly raised when running an operation
  that reads a `tf.Variable`
  before it has been initialized.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates a `FailedPreconditionError`."""
    super(FailedPreconditionError, self).__init__(node_def, op, message,
                                                  FAILED_PRECONDITION)


class AbortedError(OpError):
  """The operation was aborted, typically due to a concurrent action.

  For example, running a
  `tf.QueueBase.enqueue`
  operation may raise `AbortedError` if a
  `tf.QueueBase.close` operation
  previously ran.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates an `AbortedError`."""
    super(AbortedError, self).__init__(node_def, op, message, ABORTED)


class OutOfRangeError(OpError):
  """Raised when an operation iterates past the valid input range.

  This exception is raised in "end-of-file" conditions, such as when a
  `tf.QueueBase.dequeue`
  operation is blocked on an empty queue, and a
  `tf.QueueBase.close`
  operation executes.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates an `OutOfRangeError`."""
    super(OutOfRangeError, self).__init__(node_def, op, message,
                                          OUT_OF_RANGE)


class UnimplementedError(OpError):
  """Raised when an operation has not been implemented.

  Some operations may raise this error when passed otherwise-valid
  arguments that it does not currently support. For example, running
  the `tf.nn.max_pool2d` operation
  would raise this error if pooling was requested on the batch dimension,
  because this is not yet supported.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates an `UnimplementedError`."""
    super(UnimplementedError, self).__init__(node_def, op, message,
                                             UNIMPLEMENTED)


class InternalError(OpError):
  """Raised when the system experiences an internal error.

  This exception is raised when some invariant expected by the runtime
  has been broken. Catching this exception is not recommended.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates an `InternalError`."""
    super(InternalError, self).__init__(node_def, op, message, INTERNAL)


class UnavailableError(OpError):
  """Raised when the runtime is currently unavailable.

  This exception is not currently used.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates an `UnavailableError`."""
    super(UnavailableError, self).__init__(node_def, op, message,
                                           UNAVAILABLE)


class DataLossError(OpError):
  """Raised when unrecoverable data loss or corruption is encountered.

  For example, this may be raised by running a
  `tf.WholeFileReader.read`
  operation, if the file is truncated while it is being read.

  @@__init__
  """

  def __init__(self, node_def, op, message):
    """Creates a `DataLossError`."""
    super(DataLossError, self).__init__(node_def, op, message, DATA_LOSS)


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
  return _CODE_TO_EXCEPTION_CLASS[error_code]
