import pytest
from pytest_localserver.http import WSGIServer
import sys
import io
import requests
import json

import llm

API_BASE_TEST = "http://localhost:60412/v1"
TEST_RET_MSG = "The 2020 World Series was played in Texas at Globe Life Field in Arlington."


def api_mock():
    def simple_app(environ, start_response):
        status = "200 OK"
        response_headers = [("Content-type", "application/json")]
        write = start_response(status, response_headers)

        ret = (json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "index": 0,
                            "message": {
                                "content": TEST_RET_MSG,
                                "role": "assistant",
                            },
                            "logprobs": None,
                        }
                    ],
                    "created": 1677664795,
                    "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
                    "model": "gpt-3.5-turbo-0613",
                    "object": "chat.completion",
                    "usage": {
                        "completion_tokens": 17,
                        "prompt_tokens": 57,
                        "total_tokens": 74,
                    },
                }
            )
            + "\n").encode()

        write(ret)

        return []

    return simple_app


@pytest.fixture
def testserver(request):
    server = WSGIServer(application=api_mock())
    server.start()
    request.addfinalizer(server.stop)
    return server


# @pytest.mark.parametrize(
#     "input_str, expected_output",
#     [
#         ("test: count from 0 to 3", "generated_text: 3.0"),
#         # ("test: count from 3 to 6", "generated_text: 6.0"),
#         # ("test: count from 6 to 9", "generated_text: 9.0"),
#     ],
# )

@pytest.mark.usefixtures("testserver")
def test_rate_limits_0(testserver):
    # old_stdout = sys.stdout
    # sys.stdout = io.StringIO()

    ret = llm.llm_chat(
        [dict(role="user", content="test")],
        api_base=testserver.url+"/v1",
        model="llama3-8b-8192",
        rate_limit=70,
        rate_limit_period=10,
        log_rate_limiting=True,
        debug_llm_api=False,
    )

    assert ret == TEST_RET_MSG
    
    # captured = sys.stdout.getvalue().strip()
    # sys.stdout = old_stdout
    # assert expected_output in captured


# @pytest.mark.usefixtures("mock_server")
# @pytest.mark.parametrize("input_str, expected_output", [
#     ("test: count from 0 to 3", "generated_text: 3.0"),
#     ("test: count from 3 to 6", "generated_text: 6.0"),
#     ("test: count from 6 to 9", "generated_text: 9.0"),
# ])
# def test_rate_limits_1(api_base, capsys, expected_output):
#     old_stdout = sys.stdout
#     sys.stdout = io.StringIO()
#     test_str = """
#     Issues

#     Inconsistent rate limiting: The code only checks for rate limiting when rate_limit is not None. This means that the rate limiter will not be enforced if rate_limit is set to None. This can lead to unexpected behavior and may cause issues with the API service.
#     Incorrect calculation of total_tokens_used: The code calculates total_tokens_used by summing up the tokens used in the current window. However, this calculation is incorrect because it doesn't take into account the tokens used in previous windows. This can lead to inaccurate rate limiting.
#     Inefficient use of rate_limiter_state: The code stores the entire history of token usage in rate_limiter_state, which can lead to high memory usage and performance issues. A more efficient approach would be to store only the tokens used in the current window.
#     Wait time calculation: The wait time calculation is based on the time it takes to refill the token bucket, but it doesn't take into account the time it takes to process the current request. This can lead to inaccurate wait times.
#     Lack of consideration for concurrent requests: The code doesn't consider the case where multiple requests are made concurrently. This can lead to inaccurate rate limiting and potential starvation of requests.
#     Suggestions Use a more robust data structure: Instead of using a list to store rate_limiter_state, consider using a more efficient data structure like a heap or a queue to store the tokens used in the current window.
#     Implement a sliding window algorithm: Instead of storing the entire history of token usage, implement a sliding window algorithm that only considers the tokens used in the current window.
#     Use a more accurate wait time calculation: Consider using a more accurate wait time calculation that takes into account the time it takes to process the current request.
#     Handle concurrent requests: Consider using a lock or a semaphore to handle concurrent requests and ensure accurate rate limiting.
#     Add more logging and monitoring: Add more logging and monitoring to track the rate limiter's performance and identify potential issues.
#     """

#     llm.llm_chat([dict(role="user", content=test_str)], model="llama3-8b-8192", rate_limit=70, rate_limit_period=10, log_rate_limiting=True, return_stats=True, api_base=api_base)
#     time.sleep(3)
#     captured = sys.stdout.getvalue().strip()
#     sys.stdout = old_stdout
#     assert expected_output in captured
