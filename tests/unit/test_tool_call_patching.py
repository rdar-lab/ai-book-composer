from langchain_core.messages import AIMessage

from src.ai_book_composer.llm import ToolFixer


class TestCallToolPatching:
    def test_tool_call_is_patched(self):
        tool_call_response = AIMessage(
            content='<tool_call>\n{"name": "get_file_content", "arguments": {"file_name": "2307.06435v10.pdf", "length": 5000, "start_char": 0}}')
        patched_msg = ToolFixer._patch_tool_call(tool_call_response, [])
        assert len(patched_msg.tool_calls) == 1
        tool_call = patched_msg.tool_calls[0]
        if 'id' in tool_call:
            del tool_call['id']  # Remove dynamic id for comparison
        expected_tool_call = {'args': {'file_name': '2307.06435v10.pdf', 'length': 5000, 'start_char': 0},
                              'name': 'get_file_content',
                              'type': 'tool_call'}
        assert tool_call == expected_tool_call

        # Now check duplication handling
        tool_call_response = AIMessage(
            content='<tool_call>\n{"name": "get_file_content", "arguments": {"file_name": "2307.06435v10.pdf", "length": 5000, "start_char": 0}}')
        patched_msg = ToolFixer._patch_tool_call(tool_call_response, [patched_msg])
        assert patched_msg.content == ''
        assert len(patched_msg.tool_calls) == 1
        tool_call = patched_msg.tool_calls[0]
        if 'id' in tool_call:
            del tool_call['id']  # Remove dynamic id for comparison
        expected_tool_call = {'args': {'message': "You just attempted to call 'get_file_content' with args "
                                                  "{'file_name': '2307.06435v10.pdf', 'length': 5000, "
                                                  "'start_char': 0}, but this exact call was already "
                                                  'executed in the history. Check previous outputs and '
                                                  'advance to the next step.'},
                              'name': 'system_notification',
                              'type': 'tool_call'}
        assert tool_call == expected_tool_call
