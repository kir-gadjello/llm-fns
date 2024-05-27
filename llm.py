import os
import requests
import time
import json
from threading import Lock

from .pmemo import memo_cache_get, memo_cache_set, universal_hash

rate_limiter_state = {}
rate_limiter_lock = Lock()


def UserMsg(s):
    return dict(role="user", content=s)

def SystemMsg(s):
    return dict(role="system", content=s)

def AiMsg(s):
    return dict(role="assistant", content=s)

def cleanup_rate_limiter_state(key, rate_limit_period):
    global rate_limiter_state
    current_time = time.time()
    if key not in rate_limiter_state:
        rate_limiter_state[key] = []
    rate_limiter_state[key] = [
        (t, n)
        for t, n in rate_limiter_state[key]
        if current_time - t < rate_limit_period
    ]
    total_tokens_used = sum(n for t, n in rate_limiter_state[key])
    return total_tokens_used


def wait_ratelimit(key, rate_limit, rate_limit_period, msg_len, log_rate_limiting):
    global rate_limiter_state
    with rate_limiter_lock:
        if key not in rate_limiter_state:
            rate_limiter_state[key] = []

        current_time = time.time()
        total_tokens_used = cleanup_rate_limiter_state(key, rate_limit_period)

        if msg_len > rate_limit and len(rate_limiter_state[key]):
            wait_time = min(
                rate_limit_period,
                max(
                    0,
                    rate_limiter_state[key][-1][0] - (current_time - rate_limit_period),
                ),
            )
            if log_rate_limiting:
                print(
                    f"LLM API RATE LIMITS: n_msg_tokens > rate_limit, Waiting for {round(wait_time, 2)}s to clean the window"
                )
            time.sleep(wait_time)
        else:
            while total_tokens_used + msg_len > rate_limit and len(
                rate_limiter_state[key]
            ):
                token_sum = 0
                for i, (t, n) in enumerate(rate_limiter_state[key]):
                    token_sum += n
                    if msg_len >= (rate_limit - token_sum):
                        wait_time = min(
                            rate_limit_period,
                            max(0, t - (current_time - rate_limit_period)),
                        )
                        if log_rate_limiting:
                            print(
                                f"LLM API RATE LIMITS: Waiting for {round(wait_time, 2)}s"
                            )
                        time.sleep(wait_time)
                        total_tokens_used = cleanup_rate_limiter_state(
                            key, rate_limit_period
                        )
                        break


def llm_chat(
    messages_or_msg,
    model="gpt-3.5-turbo",
    seed=0,
    temperature=0,
    postprocess=None,
    api_key=None,
    api_base=None,
    rate_limit=None,  # total tokens per rate_limit_period
    rate_limit_period=60,
    tokenizer=lambda s: len(s) / 4,
    log_rate_limiting=False,
    n_max_retries=0,  # new argument
    max_timeout=10,
    return_stats=False,
    debug_llm_api=False,
    use_cache=False,
    cache_dir=None,
    cache_tag=None,
    **kwargs,
):
    global rate_limiter_state

    debug_llm_api = debug_llm_api or os.environ.get("DEBUG_LLM_API")

    if isinstance(messages_or_msg, str):
        messages = [dict(role="user", content=messages_or_msg)]
    else:
        messages = messages_or_msg

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    if api_key is None and api_base and api_base.find("api.openai.com") > -1:
        raise ValueError("Must provide OpenAI API key")

    api_base = api_base if api_base is not None else os.environ.get("OPENAI_API_BASE")

    url = api_base
    url = url.rstrip("/")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    data = {
        "model": model,
        "seed": seed,
        "temperature": temperature,
        "messages": [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ],
        **kwargs,
    }

    resp = None
    cached = False

    if use_cache:
        resp = memo_cache_get(data, tag=cache_tag, memo_dir=cache_dir)
        cached = resp is not None
        t0 = time.time()

    if resp is None:
        if rate_limit is not None:
            key = universal_hash((api_base, api_key, model))
            msg_len = sum(map(tokenizer, map(lambda msg: msg["content"], messages)))
            wait_ratelimit(
                key, rate_limit, rate_limit_period, msg_len, log_rate_limiting
            )

        chat_url = f"{url}/chat/completions"

        timeout = 0.5
        for attempt in range(n_max_retries + 1):
            try:
                if debug_llm_api and attempt == 0:
                    print(f"HTTP POST >>> {chat_url} BODY:", json.dumps(data, indent=2))
                t0 = time.time()
                response = requests.post(chat_url, json=data, headers=headers)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt < n_max_retries:
                    print(
                        f"HTTP POST REQUEST FAILED: ({e}) (attempt {attempt+1}/{n_max_retries+1}), RETRYING..."
                    )
                    time.sleep(timeout)
                    timeout *= 2
                    timeout = min(max_timeout, timeout)
                else:
                    raise

        resp = response.json()

        if use_cache:
            memo_cache_set(data, resp, tag=cache_tag, memo_dir=cache_dir)

    if debug_llm_api:
        dt = round(time.time() - t0, 2)
        cachemsg = "<LOCAL CACHE HIT> " if cached else ""
        status_code = "200 OK" if cached else response.status_code
        print(
            f"{cachemsg}HTTP POST @ TIME={dt}s <<< CODE={status_code}, BODY:",
            json.dumps(resp, indent=2),
        )

    content = resp["choices"][0]["message"]["content"]
    usage = resp["usage"]["total_tokens"]

    if rate_limit is not None:
        with rate_limiter_lock:
            current_time = time.time()
            rate_limiter_state[key].append((current_time, usage))
            cleanup_rate_limiter_state(key, rate_limit_period)

    if postprocess:
        content = postprocess(content)

    if return_stats:
        return content, resp["usage"]

    return content


# only llama.cpp http param schema is supported for now
def llm_base_complete(
    prompt,
    model="gpt-3.5-turbo",
    seed=0,
    temperature=0,
    postprocess=None,
    api_key=None,
    api_base=None,
    stops=[],
    max_tokens=100,
    **kwargs,
):
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None and api_base and api_base.find("api.openai.com") > -1:
        raise ValueError("Must provide OpenAI API key")

    url = api_base if api_base is not None else os.environ.get("OPENAI_API_BASE")

    url = url.rstrip("/")

    if url.endswith("v1"):
        url = url[:-2]

    url = url.rstrip("/")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "model": model,
        "seed": seed,
        "temperature": temperature,
        "stop": stops,
        "n_predict": max_tokens,
        **kwargs,
    }
    if os.environ.get("DEBUG_LLM_API"):
        print(json.dumps(data, indent=2))

    response = requests.post(f"{url}/completion", json=data, headers=headers)
    response.raise_for_status()
    content = response.json()["content"]

    if postprocess:
        content = postprocess(content)

    return content


def embed_text(
    text: str,
    api_base: str,
    model: str = "text-embedding-ada-002",
    api_key=None,
    url="/embeddings",
) -> list:
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = {"input": text, "model": model, "encoding_format": "float"}
    response = requests.post(
        api_base.rstrip("/") + "/" + url.lstrip("/"), headers=headers, json=data
    )

    if response.status_code != 200:
        raise Exception(f"API error: {response.text}")

    response_json = response.json()
    if "error" in response_json:
        raise Exception(f"API error: {response_json['error']}")

    embedding = response_json["data"][0]["embedding"]
    return embedding


CLASSIFIER_SYSTEM_PROMPT = """You are an advanced text classification AI, you focus only on answering the yes/no questions about a piece of text user is going to give you.

# Input format
You will receive the following fields in the user request:
Input: "..." - an input text for the AI text classifier system to analyze carefully and classify appropriately
Question: "..." - a yes/no question for the AI text classifier system to answer about the input text.

# IMPORTANT!
Your one and only task is to use your intelligence to output either \'Answer: yes\' or \'Answer: no\'
"""

CLASSIFIER_SYSTEM_PROMPT_TRIPLEQUOTE = """You are an advanced text classification AI, you focus only on answering the yes/no questions about a piece of text user is going to give you.

# Input format
You will receive the following fields in the user request:
Input: \"\"\"...\"\"\" - an input text for the AI text classifier system to analyze carefully and classify appropriately
Question: "..." - a yes/no question for the AI text classifier system to answer about the input text.

# IMPORTANT!
Your one and only task is to use your intelligence to output either \'Answer: yes\' or \'Answer: no\'
"""

boolean_json_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "string",
    "enum": ["Answer: yes", "Answer: no"],
}


def llm_as_binary_text_classifier(
    llm_chat_api,
    prompt="Answer the following question about this input text:\n",
    system_prompt=CLASSIFIER_SYSTEM_PROMPT,
    quoting=None,
):
    def llm_classifier(text, question=None, questions=None):
        answers = []

        if isinstance(question, str):
            questions = [question]

        for question in questions:
            qtext = f'"{text}"' if not quoting else f"{quoting}{text}{quoting}"
            task = f'{prompt}\nInput: {qtext}\nQuestion: "{question}"'

            llm_response = llm_chat_api(
                [
                    dict(role="system", content=system_prompt),
                    dict(role="user", content=task),
                ],
                json_schema=boolean_json_schema,
            )

            ret = None
            if llm_response.find("yes") > -1:
                ret = True
            if llm_response.find("no") > -1:
                ret = False
            answers.append(ret)

        if isinstance(question, str):
            return answers[0]

        return answers

    return llm_classifier


def simple_xml_chat_tpl(msgs=[], append_gen_suffix=False, get_stops=False):
    if get_stops:
        return "</message>"

    ret = "<chat>\n"

    for m in msgs:
        role = m["role"]
        content = m["content"]
        ret.append(f'<message from="{role}">{content.rstrip()}</message>\n')

    if append_gen_suffix:
        ret += '<message from="assistant">'

    return ret


def make_llm_chat_api_from_completion_api(
    llm_base_complete,
    chat_template_fn=simple_xml_chat_tpl,
    preprompt="",
    prefix_history=[],
    prefix_history_fn=None,
    api_key=None,
    api_base=None,
    **sys_kwargs,
):
    kwargs = {**sys_kwargs, **dict(api_key=api_key, api_base=api_base)}

    stops = chat_template_fn(get_stops=True)

    def llm_chat(
        messages,
        model=None,
        seed=0,
        temperature=0,
        postprocess=None,
        api_key=None,
        api_base=None,
        **user_kwargs,
    ):
        if callable(prefix_history_fn):
            messages = prefix_history_fn(messages) + messages
        elif prefix_history is not None:
            messages = prefix_history + messages

        prompt = f"{preprompt}{chat_template_fn(messages, append_gen_suffix=True)}"

        llm_resp = llm_base_complete(prompt, stops=stops, **{**user_kwargs, **kwargs})

        if postprocess:
            content = postprocess(llm_resp)

        return content

    return llm_chat
