import streamlit as st
from openai import OpenAI
import json
import time

openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
model = "gpt-4o-mini"
scenario_file = "dialogue-session/counselor_scenario.json"

# カウンセラーエージェントの発話生成関数
def generate_counselor_message(counselor_scenario_message, dialogue_history, turn, scenario_data):
    counselor_message_prompt = f"""
# 命令書：
あなたは優秀なカウンセラーエージェントです。
以下の制約条件と発話シナリオ、対話履歴をもとに発話を生成してください。

# 制約条件：
- 基本的に発話シナリオに沿って、自然な発話を生成する。
- 患者から質問があった場合は発話シナリオは気にせず、簡潔に回答する。
- 患者が困り事、状況、気分、考えを述べた場合は、発話の冒頭で患者の返答に対する繰り返し（言い換え）や共感的な声かけを1文で簡潔に行う。
- 発話シナリオに含まれる説明や具体例は省略しない。
- 発話シナリオに含まれない質問や提案はしない。
- 指示的な発話や断定的な発話はしない。

# 今回のターン{turn+1}の発話シナリオ：
{counselor_scenario_message}

# 発話シナリオ一覧：
{json.dumps(scenario_data, ensure_ascii=False, indent=2)}
"""
    # カウンセラーのメッセージリストを更新（対話履歴を更新）
    messages_for_counselor = [{"role": "system", "content": counselor_message_prompt}] + dialogue_history

    counselor_response = openai.chat.completions.create(
        model=model,
        messages=messages_for_counselor,
    )
    counselor_reply = counselor_response.choices[0].message.content.strip()
    return counselor_reply

# 生成された発話を評価する関数
def check_generated_message(scenario_data, dialogue_history, counselor_reply):
    check_prompt = f"""
# 命令書：
あなたはカウンセラーエージェントが生成した発話を管理するエージェントです。
カウンセラーエージェントは基本的に発話シナリオに沿った発話を行わなければなりませんが、患者から質問があった場合は発話シナリオは気にせず、質問に回答する必要があります。
制約条件をもとにカウンセラーエージェントが生成した発話を評価してください。

# 制約条件：
- 直前の患者発話が質問を行なっているか確認する。
- 直前の患者発話が質問を行なっている場合は発話シナリオは気にせず、簡潔に回答できているか確認する。
- 直前の患者発話が質問を行なっていない場合は生成された発話に発話シナリオの内容が全て含まれていることを確認する。
  - 発話シナリオに含まれる説明や具体例が省略されていないか確認する。
  - 発話シナリオに含まれない質問、提案、早期の対話終了をしていないか確認する。
"""
    # 評価結果はboolで返す
    check_counselor_reply = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": check_prompt
            },
            {
                "role": "user",
                "content": f"""
以下は発話シナリオ、対話履歴、カウンセラーエージェントが生成した発話です。
制約条件をもとにカウンセラーエージェントが生成した発話を評価してください。

# 発話シナリオ：
{scenario_data}

# 対話履歴：
{dialogue_history}

# カウンセラーエージェントが生成した発話：
{counselor_reply}
"""
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "check_generated_message",
                    "description": "カウンセラーエージェントが生成した発話を評価する",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "result": {"type": "boolean", "description": "カウンセラーエージェントが生成した発話を評価する"},
                            "reason": {"type": "string", "description": "なぜそのように評価したかの理由を述べる"}
                        }
                    },
                    "required": [
                        "result",
                        "reason",
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }
        ],
        tool_choice="required"
    )
    result = check_counselor_reply.choices[0].message.tool_calls[0].function.arguments
    data = json.loads(result)
    print(f"生成発話: {data["result"]}評価の理由")
    print(data["reason"])
    return data["result"]

def judge_turn_num(scenario_data, dialogue_history):
    # 評価結果はboolで返す
    check_counselor_reply = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""
# 命令書：
あなたはカウンセリングにおける発話シナリオのターンを進めるべきか判定するエージェントです。
カウンセラーエージェントは基本的に発話シナリオに沿った発話を行わなければなりませんが、患者から質問があった場合は発話シナリオは気にせず、質問に回答する必要があります。
制約条件をもとに発話シナリオのターンを進めるべきか判定してください。

# 制約条件：
- 直前の患者発話が質問を行なっているか確認する。
- 直前の患者発話が質問を行なっている場合は発話シナリオのターンを進めないと判定する。
- 直前の患者発話が質問を行なっていない場合は発話シナリオのターンを進めると判定する。
"""
            },
            {
                "role": "user",
                "content": f"""
以下は発話シナリオ、対話履歴です。
制約条件をもとに発話シナリオのターンを進めるべきか判定してください。

# 発話シナリオ：
{scenario_data}

# 対話履歴：
{dialogue_history}
"""
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "judge_turn_num",
                    "description": "発話シナリオのターンを進めるべきか判定する",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "result": {"type": "boolean", "description": "発話シナリオのターンを進めるべきか判定する"},
                            "reason": {"type": "string", "description": "なぜそのように評価したかの理由を述べる"}
                        }
                    },
                    "required": [
                        "result",
                        "reason",
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }
        ],
        tool_choice="required"
    )
    result = check_counselor_reply.choices[0].message.tool_calls[0].function.arguments
    data = json.loads(result)
    print(f"ターンを進めるか: {data["result"]}評価の理由")
    print(data["reason"])
    return data["result"]

# def judge_turn_num(scenario_data, dialogue_history):
#     # 評価結果はintで返す
#     check_counselor_reply = openai.chat.completions.create(
#         model=model,
#         messages=[
#             {
#                 "role": "system",
#                 "content": f"""
# あなたはカウンセリングが発話シナリオの何ターン目まで終えているかを判定するエージェントです。
# 発話シナリオはカウンセラーが話す予定の内容です。
# 発話シナリオと対話履歴をもとに、何ターン目の発話シナリオに対する対話まで終えているかを判定してください。
# 患者から質問があったり、患者がカウンセラーの発話を理解できていない場合は、その発話シナリオはまだ終えていないと判定してください。
# """
#             },
#             {
#                 "role": "user",
#                 "content": f"""
# 以下は発話シナリオ、対話履歴です。
# カウンセリングが発話シナリオの何ターン目まで終えているかを判定してください。

# # 発話シナリオ：
# {scenario_data}

# # 対話履歴：
# {dialogue_history}
# """
#             }
#         ],
#         tools=[
#             {
#                 "type": "function",
#                 "function": {
#                     "name": "judge_turn_num",
#                     "description": "カウンセリングが発話シナリオの何ターン目まで終えているかを判定する",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "result": {"type": "integer", "minimum": 1, "maximum": 21,  "description": "カウンセリングが発話シナリオの何ターン目まで終えているかを判定する"},
#                         }
#                     },
#                     "required": [
#                         "result",
#                     ],
#                     "additionalProperties": False
#                 },
#                 "strict": True
#             }
#         ],
#         tool_choice="required"
#     )
#     result = check_counselor_reply.choices[0].message.tool_calls[0].function.arguments
#     data = json.loads(result)
#     return data["result"]

# ストリーム表示を行う関数
def stream_counselor_reply(counselor_reply):
    for chunk in counselor_reply:
        yield chunk
        time.sleep(0.02)

# 対話セッション
if st.session_state.current_page == "dialogue":
    st.title("対話セッション")

    if "counselor_turn" not in st.session_state:
        st.session_state.counselor_turn = 0

    if "messages_for_counselor" not in st.session_state:
        st.session_state.messages_for_counselor = []

    # どちらが発話したか
    if "speaker" not in st.session_state:
        st.session_state.speaker = "counselor"

    with open(scenario_file, "r") as f:
        scenario_data = json.load(f)["counselor_scenario"]

    # サイドバーにターン進捗を表示
    with st.sidebar:
        st.markdown(f"### 実験の進度")
        st.progress(2 / 5)

    # 対話履歴を表示し続ける
    for dialogue_history in st.session_state.dialogue_history:
        with st.chat_message(dialogue_history["role"]):
            st.markdown(dialogue_history["content"])

    # 現在のターンのカウンセラーエージェントの発話を生成・表示
    if st.session_state.speaker == "counselor":
        # まだ表示されていない発話のみをストリーミング表示する
        # if len(st.session_state.messages_for_counselor) == st.session_state.counselor_turn:
        counselor_scenario_message = scenario_data[st.session_state.counselor_turn]["counselor_message"]

        # 1ターン目はシナリオ通りの発話を使用
        if st.session_state.counselor_turn == 0:
            # 表示を遅らせる
            time.sleep(2)
            counselor_reply = counselor_scenario_message
        # 2ターン目以降はカウンセラーエージェントの発話を生成
        else:
            # 3回までは生成する
            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                # 直前の患者の発話
                previous_user_message = st.session_state.dialogue_history[-1]["content"]

                counselor_reply = generate_counselor_message(counselor_scenario_message, st.session_state.dialogue_history, st.session_state.counselor_turn, scenario_data)
                # チェックはboolが返ってくるまで何回でも行う
                check_result = None
                while not isinstance(check_result, bool):
                    try:
                        check_result = check_generated_message(scenario_data, st.session_state.dialogue_history, counselor_reply)
                    except Exception as e:
                        print(f"チェックエラーが発生しました。再試行します: {e}")
                if check_result:
                    break
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"ターン{st.session_state.counselor_turn+1}: 発話がシナリオから逸脱しています。再生成します。（{retry_count}/{max_retries}）")
                        st.session_state.deviation_history.append(f"ターン{st.session_state.counselor_turn+1}: 発話がシナリオから逸脱しています。再生成します。（{retry_count}/{max_retries}）")
                        st.session_state.deviation_history.append(f"逸脱と判断された発話：{counselor_reply}")
                        print(f"逸脱と判断された発話：{counselor_reply}")
                    else:
                        # 2回生成しても発話シナリオから逸脱していた場合は、シナリオ通りの発話を使用
                        print(f"❌ ターン{st.session_state.counselor_turn+1}: 最大再生成回数に達しました。シナリオ通りの発話を使用します。")
                        st.session_state.deviation_history.append(f"❌ ターン{st.session_state.counselor_turn+1}: 最大再生成回数に達しました。シナリオ通りの発話を使用します。")
                        st.session_state.deviation_history.append(f"逸脱と判断された発話：{counselor_reply}")
                        print(f"逸脱と判断された発話：{counselor_reply}")
                        counselor_reply = counselor_scenario_message

        # カウンセラーエージェントの発話をストリーム表示
        with st.chat_message("assistant"):
            st.write_stream(stream_counselor_reply(counselor_reply))

        # 対話履歴に追加
        st.session_state.dialogue_history.append({"role": "assistant", "content": counselor_reply})
        st.session_state.messages_for_counselor.append({"role": "assistant", "content": counselor_reply})

        if st.session_state.counselor_turn < len(scenario_data) - 1:
            st.session_state.speaker = "client"
        else:
            time.sleep(1)
            st.success("これで対話セッションは終了です。")
            if st.button("「認知の変化の回答」に進む"):
                st.rerun()

    # 被験者の入力（23ターン目は入力を求めない）
    if st.session_state.speaker == "client":
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("あなたの返答を入力してください", key="chat_input")
            submitted = st.form_submit_button("送信")

        if submitted and user_input:
            st.session_state.dialogue_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            # judge_result = judge_turn_num(scenario_data, st.session_state.dialogue_history)
            # if judge_result:
            #     st.session_state.counselor_turn += 1
            #     print("ターンを進める")
            #     print(st.session_state.counselor_turn + 1)
            # else:
            #     print("ターンを進めない")
            #     print(st.session_state.counselor_turn)

            judge_result = None
            while not isinstance(judge_result, bool):
                try:
                    judge_result = judge_turn_num(scenario_data, st.session_state.dialogue_history)
                except Exception as e:
                    print(f"ジャッジエラーが発生しました。再試行します: {e}")
            if judge_result:
                st.session_state.counselor_turn += 1
                print("ターンを進める")
            else:
                print("ターンを進めない")

            print(f"現在のターン：{st.session_state.counselor_turn + 1}")

            # st.session_state.counselor_turn += 1
            st.session_state.speaker = "counselor"
            st.rerun()

    # # 23ターン終了
    # if st.session_state.counselor_turn:
    #     time.sleep(1)
    #     st.success("これで対話セッションは終了です。")
    #     if st.button("「認知の変化の回答」に進む"):
    #         # st.session_state.current_page = "cc_immediate"
    #         st.rerun()

else:
    st.session_state.current_page = "description"
    st.rerun()