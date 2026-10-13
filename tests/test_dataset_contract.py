import pytest
from core.dataset_contract import validate_sign_name, validate_subject_id, validate_session_id


def test_special_classes_and_ascii_sign_names():
    for s in ["xin_chao","cam_on","khong","REST","UNKNOWN"]:
        validate_sign_name(s)


def test_invalid_subject_is_rejected():
    with pytest.raises(ValueError): validate_subject_id("NguyenVanA")
    validate_subject_id("S01")
    validate_session_id("SES01")
