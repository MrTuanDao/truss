from pathlib import Path

import click
from jinja2 import Environment, FileSystemLoader
from truss.constants import (
    CONTROL_SERVER_CODE_DIR,
    MODEL_DOCKERFILE_NAME,
    MODEL_README_NAME,
    REQUIREMENTS_TXT_FILENAME,
    SERVER_CODE_DIR,
    SERVER_DOCKERFILE_TEMPLATE_NAME,
    SERVER_REQUIREMENTS_TXT_FILENAME,
    SHARED_SERVING_AND_TRAINING_CODE_DIR,
    SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME,
    SYSTEM_PACKAGES_TXT_FILENAME,
    TEMPLATES_DIR,
)
from truss.contexts.image_builder.image_builder import ImageBuilder
from truss.contexts.truss_context import TrussContext
from truss.patch.hash import directory_content_hash
from truss.readme_generator import generate_readme
from truss.truss_spec import TrussSpec
from truss.utils import build_truss_target_directory, copy_file_path, copy_tree_path

BUILD_SERVER_DIR_NAME = "server"
BUILD_CONTROL_SERVER_DIR_NAME = "control"


class ServingImageBuilderContext(TrussContext):
    @staticmethod
    def run(truss_dir: Path):
        return ServingImageBuilder(truss_dir)


class ServingImageBuilder(ImageBuilder):
    def __init__(self, truss_dir: Path) -> None:
        self._truss_dir = truss_dir
        self._spec = TrussSpec(truss_dir)

    @property
    def default_tag(self):
        return f"{self._spec.model_framework_name}-model:latest"

    def prepare_image_build_dir(self, build_dir: Path = None):
        """Prepare a directory for building the docker image from.

        Returns:
            docker command to build the docker image.
        """
        if build_dir is None:
            build_dir = build_truss_target_directory(self._spec.model_framework_name)
            # todo: Add a logging statement here, suggesting how to clean up the directory.

        copy_tree_path(self._spec.truss_dir, build_dir)
        copy_tree_path(
            SERVER_CODE_DIR,
            build_dir / BUILD_SERVER_DIR_NAME,
        )
        copy_tree_path(
            SHARED_SERVING_AND_TRAINING_CODE_DIR,
            build_dir
            / BUILD_SERVER_DIR_NAME
            / SHARED_SERVING_AND_TRAINING_CODE_DIR_NAME,
        )
        if self._spec.config.live_reload:
            copy_tree_path(
                CONTROL_SERVER_CODE_DIR,
                build_dir / BUILD_CONTROL_SERVER_DIR_NAME,
            )

        with (build_dir / REQUIREMENTS_TXT_FILENAME).open("w") as req_file:
            req_file.write(self._spec.requirements_txt)

        with (build_dir / SYSTEM_PACKAGES_TXT_FILENAME).open("w") as req_file:
            req_file.write(self._spec.system_packages_txt)

        data_dir_exists = (build_dir / self._spec.config.data_dir).exists()
        bundled_packages_dir_exists = (
            build_dir / self._spec.config.bundled_packages_dir
        ).exists()

        template_loader = FileSystemLoader(str(TEMPLATES_DIR))
        template_env = Environment(loader=template_loader)
        dockerfile_template = template_env.get_template(SERVER_DOCKERFILE_TEMPLATE_NAME)
        config = self._spec.config
        base_image_name = f"baseten/truss-base-{config.python_version}-{config.model_framework.value}"
        if config.resources.use_gpu:
            base_image_name = f"{base_image_name}-gpu"
        if config.live_reload:
            base_image_name = f"{base_image_name}-reload"
        tag = "test"  # todo: change to latest
        base_image_name_and_tag = f"{base_image_name}:{tag}"
        should_install_system_requirements = _file_is_not_empty(build_dir / SYSTEM_PACKAGES_TXT_FILENAME)
        should_install_requirements = _file_is_not_empty(build_dir / REQUIREMENTS_TXT_FILENAME)
        dockerfile_contents = dockerfile_template.render(
            base_image_name_and_tag=base_image_name_and_tag,
            should_install_system_requirements=should_install_system_requirements,
            should_install_requirements=should_install_requirements,
            config=config,
            data_dir_exists=data_dir_exists,
            bundled_packages_dir_exists=bundled_packages_dir_exists,
            truss_hash=directory_content_hash(self._truss_dir),
        )
        docker_file_path = build_dir / MODEL_DOCKERFILE_NAME
        with docker_file_path.open("w") as docker_file:
            docker_file.write(dockerfile_contents)

        readme_file_path = build_dir / MODEL_README_NAME
        try:
            readme_contents = generate_readme(self._spec)
            with readme_file_path.open("w") as readme_file:
                readme_file.write(readme_contents)
        except Exception as e:
            click.echo(
                click.style(
                    f"""WARNING: Auto-readme generation has failed.
                    This is probably due to a malformed config.yaml or
                    malformed examples.yaml. Error is:
                    {e}
                    """,
                    fg="yellow",
                )
            )


def _file_is_empty(path: Path, ignore_hash_style_comments: bool = True) -> bool:
    if not path.exists():
        return True

    with path.open() as file:
        for line in file.readlines():
            if ignore_hash_style_comments and _is_hash_style_comment(line):
                continue
            if line.strip() != "":
                return False

    return True


def _file_is_not_empty(path: Path, ignore_hash_style_comments: bool = True) -> bool:
    return not _file_is_empty(path, ignore_hash_style_comments)


def _is_hash_style_comment(line: str) -> bool:
    return line.lstrip().startswith("#")
