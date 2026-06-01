import pytest
import warnings

from pipegenie.exceptions import (
    PipegenieError,
    NoValidPipelines,
    NoFittedModelError,
    PipegenieWarning,
    NoValidPipelineWarning,
    warn,
)

def test_pipegenieerror_is_exception():
    err = PipegenieError("base error")
    assert isinstance(err, Exception)

def test_novalidpipelines_error_message_and_invalid_value():
    error = NoValidPipelines(
        message="No pipelines found",
        invalid_value=42
    )

    assert isinstance(error, PipegenieError)
    assert error.invalid_value == 42
    assert str(error) == "No pipelines found. Invalid value: 42"

def test_novalidpipelines_error_with_none_invalid_value():
    error = NoValidPipelines(
        message="No pipelines found",
        invalid_value=None
    )

    assert error.invalid_value is None
    assert str(error) == "No pipelines found. Invalid value: None"

def test_nofittedmodel_error_without_config_key():
    error = NoFittedModelError("Model is not fitted")

    assert isinstance(error, PipegenieError)
    assert error.config_key is None
    assert str(error) == "Model is not fitted"

def test_nofittedmodel_error_with_config_key():
    error = NoFittedModelError(
        message="Model is not fitted",
        config_key="model_name"
    )

    assert error.config_key == "model_name"
    assert str(error) == "Model is not fitted [key: model_name]"

def test_pipegeniewarning_is_warning():
    assert issubclass(PipegenieWarning, Warning)

def test_novalidpipelinewarning_is_pipegenie_warning():
    assert issubclass(NoValidPipelineWarning, PipegenieWarning)

def test_warn_emits_pipegenie_warning():
    with pytest.warns(PipegenieWarning, match="Something happened"):
        warn("Something happened")

def test_warn_emits_custom_warning_category():
    with pytest.warns(NoValidPipelineWarning, match="No valid pipeline"):
        warn(
            "No valid pipeline",
            category=NoValidPipelineWarning
        )

def test_warn_stacklevel_points_to_caller():
    with pytest.warns(PipegenieWarning) as record:
        warn("stacklevel test")

    warning = record[0]
    assert "exceptions.py" not in warning.filename
