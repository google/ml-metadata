def pytest_addoption(parser):
    parser.addoption(
        '--use_grpc_backend', action='store_true', default=False,
        help='Set this to true to use gRPC instead of sqlLite backend.'
    )
    parser.addoption(
        '--grpc_host', type=str, default=None,
        help="The gRPC host name to use when use_grpc_backed is set to 'True'"
    )
    parser.addoption(
        '--grpc_port', type=int, default=0,
        help="The gRPC port number to use when use_grpc_backed is set to 'True'"
    )
