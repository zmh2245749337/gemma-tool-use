from gemma_eval.preference_data import classify_model_error


def target(decision="no_tool", tool_call=None, missing_slots=None):
    return {
        "belief_state": [],
        "decision": decision,
        "tool_call": tool_call,
        "missing_slots": missing_slots or [],
    }


def test_preference_pair_classifies_false_tool_call():
    category, comparison = classify_model_error(
        '{"belief_state":[],"decision":"call_tool","tool_call":{"name":"query_hotel_db","arguments":{"constraints":{},"requested_fields":[]}},"missing_slots":[]}',
        target(),
    )
    assert category == "false_tool_call"
    assert comparison is not None


def test_preference_pair_skips_semantically_correct_output():
    category, comparison = classify_model_error(
        '{"belief_state":[],"decision":"ask_user","tool_call":null,"missing_slots":["目的地"]}',
        target("ask_user", missing_slots=["目的地"]),
    )
    assert category is None
    assert comparison is not None


def test_preference_pair_keeps_invalid_json_as_negative():
    category, comparison = classify_model_error("not json", target())
    assert category == "schema_invalid"
    assert comparison is None
