from scripts.build_v2_curriculum import (
    argument_item_count,
    category_rows,
    duplicate,
)


def make_row(sample_id="row-1", decision="call_tool", text="请帮我查询"):
    tool_call = None
    if decision == "call_tool":
        tool_call = {
            "name": "query_hotel_db",
            "arguments": {
                "constraints": {"价格": "300-400元", "评分": "4.5分以上"},
                "requested_fields": ["名称", "地址"],
            },
        }
    return {
        "sample_id": sample_id,
        "history": [],
        "previous_state": [],
        "current_user": text,
        "output": {
            "belief_state": [],
            "decision": decision,
            "tool_call": tool_call,
            "missing_slots": [],
        },
        "source": {"dialog_id": "demo"},
    }


def test_argument_item_count_flattens_nested_arguments():
    value = {"constraints": {"价格": "300-400元", "评分": "4.5分以上"}, "fields": ["名称"]}
    assert argument_item_count(value) == 3


def test_curriculum_categories_use_training_labels_without_mutating_them():
    no_tool = make_row("no-tool", "no_tool", "好的，谢谢")
    correction = make_row("correction", text="不是四百分，改成三百分")
    buckets = category_rows([no_tool, correction], minimum_argument_items=4, minimum_state_items=10)
    assert buckets["no_tool_boundary"] == [no_tool]
    assert buckets["condition_correction"] == [correction]
    assert buckets["parameter_dense"] == [correction]

    copied = duplicate(correction, "condition_correction", 1)
    assert copied["output"] == correction["output"]
    assert copied["sample_id"] != correction["sample_id"]
    assert copied["source"]["augmentation"]["labels_modified"] is False
