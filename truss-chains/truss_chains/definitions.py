# TODO: this file contains too much implementation -> restructure.

import abc
import logging
import os
import traceback
from types import GenericAlias
from typing import Any, ClassVar, Generic, Mapping, Optional, Type, TypeVar, Union

import pydantic
from pydantic import generics
from truss import truss_config

UserConfigT = TypeVar("UserConfigT", bound=Optional[pydantic.BaseModel])

BASETEN_API_SECRET_NAME = "baseten_chain_api_key"
SECRET_DUMMY = "***"
TRUSS_CONFIG_CHAINS_KEY = "chains_metadata"

ENDPOINT_METHOD_NAME = "run"  # Referring to Chainlet method name exposed as endpoint.
# Below arg names must correspond to `definitions.ABCChainlet`.
CONTEXT_ARG_NAME = "context"  # Referring to Chainlets `__init__` signature.
SELF_ARG_NAME = "self"

GENERATED_CODE_DIR = ".chains_generated"
PREDICT_ENDPOINT_NAME = "/predict"
chainlet_MODULE = "Chainlet"
STUB_TYPE_MODULE = "stub_types"
STUB_CLS_SUFFIX = "Stub"


class APIDefinitionError(TypeError):
    """Raised when user-defined Chainlets do not adhere to API constraints."""


class MissingDependencyError(TypeError):
    """Raised when a needed resource could not be found or is not defined."""


class UsageError(Exception):
    """Raised when components are not used the expected way at runtime."""


class AbsPath:
    _abs_file_path: str
    _creating_module: str
    _original_path: str

    def __init__(
        self, abs_file_path: str, creating_module: str, original_path: str
    ) -> None:
        self._abs_file_path = abs_file_path
        self._creating_module = creating_module
        self._original_path = original_path

    def raise_if_not_exists(self) -> None:
        if not os.path.isfile(self._abs_file_path):
            raise MissingDependencyError(
                f"With the file path `{self._original_path}` an absolute path relative "
                f"to the calling module `{self._creating_module}` was created, "
                f"resulting `{self._abs_file_path}` - but no file was found."
            )

    @property
    def abs_path(self) -> str:
        if self._abs_file_path != self._original_path:
            logging.info(
                f"Using abs path `{self._abs_file_path}` for path specified as "
                f"`{self._original_path}` (in `{self._creating_module}`)."
            )
        return self._abs_file_path


class DockerImageSpec(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True

    # TODO: this is not stable yet and might change or refer back to truss.
    base_image: str = "python:3.11-slim"
    pip_requirements_file: Optional[AbsPath] = None
    pip_requirements: list[str] = []
    apt_requirements: list[str] = []


class DockerImage:
    """Builder to create ImageSpec."""

    _spec: DockerImageSpec

    def __init__(self) -> None:
        self._spec = DockerImageSpec()

    def base_image(self, base_image: str) -> "DockerImage":
        self._spec.base_image = base_image
        return self

    def pip_requirements_file(self, file_path: AbsPath) -> "DockerImage":
        self._spec.pip_requirements_file = file_path
        return self

    def pip_requirements(self, requirements: list[str]) -> "DockerImage":
        self._spec.pip_requirements = requirements
        return self

    def apt_requirements(self, requirements: list[str]) -> "DockerImage":
        self._spec.apt_requirements = requirements
        return self

    def get_spec(self) -> DockerImageSpec:
        return self._spec.copy(deep=True)


class ComputeSpec(pydantic.BaseModel):
    # TODO: this is not stable yet and might change or refer back to truss.
    cpu: int = 1
    memory: str = "2Gi"
    gpu: truss_config.AcceleratorSpec = truss_config.AcceleratorSpec()


class Compute:
    """Builder to create ComputeSpec."""

    _spec: ComputeSpec

    def __init__(self) -> None:
        self._spec = ComputeSpec()

    def cpu(self, cpu: int) -> "Compute":
        self._spec.cpu = cpu
        return self

    def memory(self, memory: str) -> "Compute":
        self._spec.memory = memory
        return self

    def gpu(
        self, kind: Union[str, truss_config.Accelerator], count: int = 1
    ) -> "Compute":
        self._spec.gpu = truss_config.AcceleratorSpec(
            accelerator=truss_config.Accelerator(kind), count=count
        )
        return self

    def get_spec(self) -> ComputeSpec:
        return self._spec.copy(deep=True)


class AssetSpec(pydantic.BaseModel):
    # TODO: this is not stable yet and might change or refer back to truss.
    secrets: dict[str, str] = {}
    cached: list[Any] = []


class Assets:
    """Builder to create asset spec."""

    _spec: AssetSpec

    def __init__(self) -> None:
        self._spec = AssetSpec()

    def add_secret(self, key: str) -> "Assets":
        # Actual value is provided in deployment.
        self._spec.secrets[key] = SECRET_DUMMY
        return self

    def cached(self, value: list[Any]) -> "Assets":
        self._spec.cached = value
        return self

    def get_spec(self) -> AssetSpec:
        return self._spec.copy(deep=True)


class RemoteConfig(pydantic.BaseModel):
    """Bundles config values needed to deploy a Chainlet."""

    class Config:
        arbitrary_types_allowed = True

    docker_image: DockerImage
    compute: Compute = Compute()
    assets: Assets = Assets()
    name: Optional[str] = None

    def get_docker_image_spec(self) -> DockerImageSpec:
        return self.docker_image.get_spec()

    def get_compute_spec(self) -> ComputeSpec:
        return self.compute.get_spec()

    def get_asset_spec(self) -> AssetSpec:
        return self.assets.get_spec()


class ServiceDescriptor(pydantic.BaseModel):
    name: str
    predict_url: str


class DeploymentContext(generics.GenericModel, Generic[UserConfigT]):
    """Bundles config values and resources needed to instantiate Chainlets."""

    class Config:
        arbitrary_types_allowed = True

    user_config: UserConfigT = pydantic.Field(default=None)
    chainlet_to_service: Mapping[str, ServiceDescriptor] = {}
    # secrets: Optional[secrets_resolver.Secrets] = None
    # TODO: above type results in `truss.server.shared.secrets_resolver.Secrets`
    #   due to the templating, at runtime the object passed will be from
    #   `shared.secrets_resolver` and give pydantic validation error.
    secrets: Optional[Any] = None

    def get_service_descriptor(self, chainlet_name: str) -> ServiceDescriptor:
        if chainlet_name not in self.chainlet_to_service:
            raise MissingDependencyError(f"{chainlet_name}")
        return self.chainlet_to_service[chainlet_name]

    def get_baseten_api_key(self) -> str:
        if self.secrets is None:
            raise UsageError(f"Secrets not set in `{self.__class__.__name__}` object.")
        error_msg = (
            "For using chains, it is required to setup a an API key with name "
            f"`{BASETEN_API_SECRET_NAME}` on baseten to allow chain Chainlet to "
            "call other Chainlets. For local execution, secrets can be provided "
            "to `run_local`."
        )
        if BASETEN_API_SECRET_NAME not in self.secrets:
            raise MissingDependencyError(error_msg)

        api_key = self.secrets[BASETEN_API_SECRET_NAME]
        if api_key == SECRET_DUMMY:
            raise MissingDependencyError(
                f"{error_msg}. Retrieved dummy value of `{api_key}`."
            )
        return api_key


class TrussMetadata(generics.GenericModel, Generic[UserConfigT]):
    """Plugin for the truss config (in config["model_metadata"]["chains_metadata"])."""

    user_config: UserConfigT = pydantic.Field(default=None)
    chainlet_to_service: Mapping[str, ServiceDescriptor] = {}


class ABCChainlet(Generic[UserConfigT], abc.ABC):
    remote_config: ClassVar[RemoteConfig]
    default_user_config: ClassVar[Optional[pydantic.BaseModel]] = None
    _init_is_patched: ClassVar[bool] = False

    _context: DeploymentContext[UserConfigT]

    @abc.abstractmethod
    def __init__(self, context: DeploymentContext[UserConfigT]) -> None:
        ...

    # Cannot add this abstract method to API, because we want to allow arbitrary
    # arg/kwarg names and specifying any function signature here would give type errors
    # @abc.abstractmethod
    # def run(self, *args, **kwargs) -> Any: ...

    @property
    @abc.abstractmethod
    def user_config(self) -> UserConfigT:
        ...


class TypeDescriptor(pydantic.BaseModel):
    """For describing I/O types of Chainlets.

    Example:
        repr(raw): <class 'user_package.shared_chainlet.SplitTextInput'>"
        as_src_str(): 'SplitTextInput'

    -> Source string, without further qualification, does not include any module path.
    """

    raw: Any  # The raw type annotation object (could be a type or GenericAlias).

    def as_src_str(self, qualify_pydantic_types: Optional[str] = None) -> str:
        # TODO: pydantic types will soon be handled differently.
        if self.is_pydantic and qualify_pydantic_types:
            return f"{qualify_pydantic_types}.{self.raw.__name__}"

        if isinstance(self.raw, type):
            return self.raw.__name__
        else:
            return str(self.raw)

    @property
    def is_pydantic(self) -> bool:
        return (
            isinstance(self.raw, type)
            and not isinstance(self.raw, GenericAlias)
            and issubclass(self.raw, pydantic.BaseModel)
        )


class EndpointAPIDescriptor(pydantic.BaseModel):
    name: str = ENDPOINT_METHOD_NAME
    input_names_and_types: list[tuple[str, TypeDescriptor]]
    output_types: list[TypeDescriptor]
    is_async: bool
    is_generator: bool


class ChainletAPIDescriptor(pydantic.BaseModel):
    chainlet_cls: Type[ABCChainlet]
    src_path: str
    dependencies: Mapping[str, Type[ABCChainlet]]
    endpoint: EndpointAPIDescriptor
    user_config_type: TypeDescriptor

    def __hash__(self) -> int:
        return hash(self.chainlet_cls)

    @property
    def name(self) -> str:
        return self.chainlet_cls.__name__


class StackFrame(pydantic.BaseModel):
    filename: str
    lineno: Optional[int]
    name: str
    line: Optional[str]

    @classmethod
    def from_frame_summary(cls, frame: traceback.FrameSummary):
        return cls(
            filename=frame.filename,
            lineno=frame.lineno,
            name=frame.name,
            line=frame.line,
        )

    def to_frame_summary(self) -> traceback.FrameSummary:
        return traceback.FrameSummary(
            filename=self.filename, lineno=self.lineno, name=self.name, line=self.line
        )


class RemoteErrorDetail(pydantic.BaseModel):
    remote_name: str
    exception_cls_name: str
    exception_module_name: Optional[str]
    exception_message: str
    user_stack_trace: list[StackFrame]

    def to_stack_summary(self) -> traceback.StackSummary:
        return traceback.StackSummary.from_list(
            frame.to_frame_summary() for frame in self.user_stack_trace
        )

    def format(self) -> str:
        stack = "".join(traceback.format_list(self.to_stack_summary()))
        exc_info = (
            f"\n(Exception class defined in `{self.exception_module_name}`.)"
            if self.exception_module_name
            else ""
        )
        error = (
            f"{RemoteErrorDetail.__name__} in `{self.remote_name}`\n"
            f"Traceback (most recent call last):\n"
            f"{stack}{self.exception_cls_name}: {self.exception_message}{exc_info}"
        )
        return error


class GenericRemoteException(Exception):
    ...