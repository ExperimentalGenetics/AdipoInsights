import inspect
import json
import os
import shutil
from pathlib import Path
from timeit import default_timer
from typing import Dict


class Pipeline(object):
    def __init__(self, config_path: Path, clear_existing=False):
        """
        Simple class to define an execution pipeline based on directories. It automatically skips a step if the output directory already exists.

        Args:
            config_path: path to the configuration file.
            clear_existing: whether to remove the output directories for each execution step (results in a fresh run where everything is computed from scretch).
        """
        self.clear_existing = clear_existing
        self.steps = []

        with open(config_path, 'r') as fp:
            self.config = json.load(fp)

        self.return_values = {}

    def add_step(self, function, inputs: Dict[str, Path],
                outputs: Dict[str, Path], always_compute=False) -> None:
        """
        Add an execution step to the pipeline.

        Args:
            function: The function which is executed for the step. The input and output paths are passed as named arguments to the function.
            inputs: The input paths of the function.
            outputs: The output paths of the function (will be created if necessary).
            always_compute: Whether the step should always be computed regardless of existing files (however, output directories will not be cleared).
        """
        outputs_temp = {key: output_path.with_name('running_' + output_path.name)
                        for key, output_path in outputs.items()}

        self.steps.append({
            'function': function,
            'inputs': inputs,
            'outputs': outputs,
            'outputs_temp': outputs_temp,
            'always_compute': always_compute,
        })

    def run(self) -> None:
        """
        Starts the execution of the pipeline.

        Raises:
            ValueError: If a step could not be run.
        """
        for step in self.steps:
            for input_path in step['inputs'].values():
                if input_path is not None and not input_path.exists():
                # if input_path is not None and not os.path.exists(input_path):
                    raise ValueError(
                        f'The input {input_path.name} does not exists. Cannot compute step {step["function"].__name__}. Aborting...')

            if step['always_compute']:
                for output_path in step['outputs'].values():
                    if output_path.suffix == '' and not output_path.exists():
                        output_path.mkdir(parents=True, exist_ok=True)
                self._run_function(step, 'outputs')
            else:
                if self.clear_existing:
                    for output_path in step['outputs'].values():
                        Pipeline._remove_path(output_path)

                run_step = False
                for output_path in step['outputs'].values():
                    if not output_path.exists():
                        run_step = True

                if run_step:
                    for temp_path in step['outputs_temp'].values():
                        Pipeline._remove_path(temp_path)
                        if temp_path.suffix == '':
                            temp_path.mkdir(parents=True, exist_ok=True)

                    self._run_function(step, 'outputs_temp')
                    for temp_path, output_path in zip(step['outputs_temp'].values(), step['outputs'].values()):
                        if output_path.exists():
                            Pipeline._remove_path(output_path)

                        os.rename(temp_path, output_path)
                else:
                    print(
                        f'Skipped step {self.steps.index(step) + 1} ({step["function"].__name__}) since the output data already exists')

    def _run_function(self, step: dict, outputs_name: str) -> None:
        additional_parameters = {}
        parameters = inspect.signature(step['function']).parameters

        if 'config' in parameters:
            additional_parameters['config'] = self.config

        for key, value in self.return_values.items():
            if key in parameters:
                additional_parameters[key] = value

        start = default_timer()
        return_value = step['function'](**step['inputs'], **step[outputs_name], **additional_parameters)
        end = default_timer()

        seconds = end - start
        print(
            f'Finished step {self.steps.index(step) + 1} ({step["function"].__name__}) in {int(seconds // 60):d} minutes and {seconds % 60:.2f} seconds')

        if return_value is not None:
            self.return_values = {**self.return_values, **return_value}

    @staticmethod
    def _remove_path(path: Path) -> None:
        if not path.exists():
            return

        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            raise ValueError(f'Could not delete path {path}')