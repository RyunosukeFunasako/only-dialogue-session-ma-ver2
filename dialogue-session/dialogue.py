import streamlit as st
from openai import OpenAI
import json
import time

# カウンセラーエージェントの発話生成関数
def generate_counselor_message(should_advance_turn, turn, scenario_data, counselor_scenario_message, dialogue_history):
    # 質問でない場合（ターンを進める場合）
    if should_advance_turn:
        counselor_message_prompt = f"""
# 命令書：
あなたは優秀なカウンセラーエージェントです。
以下の制約条件と発話シナリオをもとに発話を生成してください。

# 制約条件：
- 発話シナリオに沿って発話を生成する。
- 必要に応じて、発話の冒頭で患者の返答に対する繰り返し（言い換え）や共感的な声かけを1文で簡潔に行う。
- 各ターンの発話シナリオの内容は生成する発話に必ず含める。
- 発話シナリオに含まれる説明や具体例は省略しない。
- 発話シナリオに含まれない説明や質問、提案はしない。
- 指示的な発話や断定的な発話はしない。

# 今回のターン{turn}の発話シナリオ：
{counselor_scenario_message}

# 発話シナリオ一覧：
{json.dumps(scenario_data, ensure_ascii=False, indent=2)}
"""
    # 質問である場合（ターンを進めない場合）
    else:
        counselor_message_prompt = f"""
# 命令書:
あなたは優秀なカウンセラーエージェントです。
以下の制約条件をもとに発話を生成してください。

# 制約条件：
- 患者の質問に簡潔に回答する。最後には回答に満足できたか、確認の問いを1文で簡潔に行う。
- 必要に応じて、発話の冒頭で患者の返答に対する繰り返し（言い換え）や共感的な声かけを1文で簡潔に行う。
- 指示的な発話や断定的な発話はしない。
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
def check_generated_message(should_advance_turn, counselor_scenario_message, previous_user_message, counselor_reply):
    # 質問でない場合（ターンを進める場合）
    if should_advance_turn:
        check_prompt = f"""
# 命令書：
あなたはカウンセラーエージェントが生成した発話を評価するエージェントです。
制約条件をもとにカウンセラーエージェントが生成した発話を評価してください。

# 制約条件：
- カウンセラーエージェントが生成した発話を確認する。
- 発話シナリオの内容が全て含まれている場合はtrueと判定する。
- 発話シナリオの内容が全て含まれていない場合はfalseと判定する。

# 発話シナリオ：
{counselor_scenario_message}
"""
    else:
        check_prompt = f"""
# 命令書：
あなたはカウンセラーエージェントが生成した発話を評価するエージェントです。
制約条件をもとにカウンセラーエージェントが生成した発話を評価してください。

# 制約条件：
- カウンセラーエージェントが生成した発話を確認する。
- 患者の質問に簡潔に回答し、最後には確認の問いが含まれている場合はtrueと判定する。
- 患者の質問に簡潔に回答できていない、もしくは最後に確認の問いが含まれていない場合はfalseと判定する。

# 患者の質問:
{previous_user_message}
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
以下はカウンセラーエージェントが生成した発話です。
制約条件をもとにカウンセラーエージェントが生成した発話を評価してください。

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

# ターンを進めるか判定する関数
def judge_turn_advance(previous_counselor_message,previous_user_message):
    # 評価結果はboolで返す
    check_counselor_reply = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""
# 命令書：
あなたは直前の患者発話が質問であるか判定するエージェントです。
制約条件をもとに直前の患者発話が質問かどうか判定してください。
"""
            },
            {
                "role": "user",
                "content": f"""
以下は直前のカウンセラーエージェント発話、直前の患者発話です。
制約条件をもとに直前の患者発話が質問か判定してください。

# 制約条件：
- 直前のカウンセラーエージェント発話、直前の患者発話を確認し、直前の患者発話は質問なのか判断する。
- 直前の患者発話が質問でない場合はtrueと判定する。
- 直前の患者発話が質問である場合はfalseと判定する。

# 直前のカウンセラーエージェント発話:
{previous_counselor_message}

# 直前の患者発話:
{previous_user_message}
"""
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "judge_turn_advance",
                    "description": "直前の患者発話が質問か判定する",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "result": {"type": "boolean", "description": "直前の患者発話が質問か判定する"},
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

# ストリーム表示を行う関数
def stream_counselor_reply(counselor_reply):
    for chunk in counselor_reply:
        yield chunk
        time.sleep(0.02)

@st.dialog("CBTの対象")
def cbt_attention_modal():
    st.markdown("""
    対話セッションでは認知行動療法（CBT）を体験していただきます。
    CBTは「認知」に働きかけて気分を楽にする精神療法です。

    そのため、患者さんの困りごととして取り上げる内容は、「 **物事の受け取り方や考え方によって気分が変わるような悩み** 」が望ましいです。
    一方で、身体の病気や深刻な危機はCBTの直接の対象とはならず、本実験の想定範囲から外れてしまいます。

    対話セッションの中で「 **あなたが今お困りのことを簡単にお話しいただけますか？** 」という質問がありますが、上記の点を考慮してお答えください。
    また、「 **今** 」困っていることがない場合は、 **過去に困った出来事** についてお答えください。


    #### 対象となる困りごとの例：
    - 既読がついているのに返信が来ない
    - 小さなミスをすると自分はダメだと思ってしまう
    - 人前で話す時に過度に緊張してしまう

    #### 対象外となる困りごとの例：
    - 肩こりや頭痛がひどくて眠れない（身体の病気や症状）
    - 借金が返せず生活が成り立たない（経済問題）
    - 暴力やいじめを受けている（安全確保が必要な深刻な問題）
    """)

# 対話セッション
if st.session_state.current_page == "dialogue":
    st.title("対話セッション")

    openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    model = "gpt-4o-mini"
    scenario_file = "dialogue-session/counselor_scenario.json"

    if "counselor_turn" not in st.session_state:
        st.session_state.counselor_turn = 1

    if "messages_for_counselor" not in st.session_state:
        st.session_state.messages_for_counselor = []

    # ターン調整用
    if "should_advance_turn" not in st.session_state:
        st.session_state.should_advance_turn = True

    # どちらが発話したか
    if "speaker" not in st.session_state:
        st.session_state.speaker = "counselor"

    with open(scenario_file, "r") as f:
        scenario_data = json.load(f)["counselor_scenario"]

    # サイドバー
    with st.sidebar:
        st.markdown(f"### 実験の進度")
        st.progress(2 / 5)

    with st.sidebar:
        st.markdown(f"### 先ほどの画面の内容")
        if st.button("CBTの対象"):
            cbt_attention_modal()

    # 対話履歴を表示し続ける
    for dialogue_history in st.session_state.dialogue_history:
        with st.chat_message(dialogue_history["role"]):
            st.markdown(dialogue_history["content"])

    # 現在のターンのカウンセラーエージェントの発話を生成・表示
    if st.session_state.speaker == "counselor":
        counselor_scenario_message = scenario_data[st.session_state.counselor_turn - 1]["counselor_message"]

        # 1ターン目はシナリオ通りの発話を使用
        if st.session_state.counselor_turn == 1:
            # 表示を遅らせる
            time.sleep(2)
            counselor_reply = counselor_scenario_message
        # 2ターン目以降はカウンセラーエージェントの発話を生成
        else:
            # 3回までは生成する
            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                # 直前のカウンセラーエージェントの発話
                previous_counselor_message = st.session_state.dialogue_history[-2]["content"]
                # 直前の患者の発話
                previous_user_reply = st.session_state.dialogue_history[-1]["content"]

                counselor_reply = generate_counselor_message(st.session_state.should_advance_turn, st.session_state.counselor_turn, scenario_data, counselor_scenario_message, st.session_state.dialogue_history)
                # チェックはboolが返ってくるまで何回でも行う
                check_result = None
                while not isinstance(check_result, bool):
                    try:
                        check_result = check_generated_message(st.session_state.should_advance_turn, counselor_scenario_message, previous_user_reply, counselor_reply)
                    except Exception as e:
                        print(f"チェックエラーが発生しました。再試行します: {e}")
                if check_result:
                    break
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"ターン{st.session_state.counselor_turn}: 発話がシナリオから逸脱しています。再生成します。（{retry_count}/{max_retries}）")
                        st.session_state.deviation_history.append(f"ターン{st.session_state.counselor_turn}: 発話がシナリオから逸脱しています。再生成します。（{retry_count}/{max_retries}）")
                        st.session_state.deviation_history.append(f"逸脱と判断された発話：{counselor_reply}")
                        print(f"逸脱と判断された発話：{counselor_reply}")
                    else:
                        # 2回再生成しても発話シナリオから逸脱していた場合は、シナリオ通りの発話を使用
                        print(f"❌ ターン{st.session_state.counselor_turn}: 最大再生成回数に達しました。シナリオ通りの発話を使用します。")
                        st.session_state.deviation_history.append(f"❌ ターン{st.session_state.counselor_turn}: 最大再生成回数に達しました。シナリオ通りの発話を使用します。")
                        st.session_state.deviation_history.append(f"逸脱と判断された発話：{counselor_reply}")
                        print(f"逸脱と判断された発話：{counselor_reply}")
                        counselor_reply = counselor_scenario_message

        # カウンセラーエージェントの発話をストリーム表示
        with st.chat_message("assistant"):
            st.write_stream(stream_counselor_reply(counselor_reply))

        # 対話履歴に追加
        st.session_state.dialogue_history.append({"role": "assistant", "content": counselor_reply})
        st.session_state.messages_for_counselor.append({"role": "assistant", "content": counselor_reply})

        if st.session_state.counselor_turn < len(scenario_data):
            st.session_state.speaker = "client"

    # 被験者の入力（21ターン目は入力を求めない）
    if st.session_state.speaker == "client":
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("あなたの返答を入力してください", key="chat_input")
            submitted = st.form_submit_button("送信")

        if submitted and user_input:
            st.session_state.dialogue_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # 直前のカウンセラーエージェントの発話
            previous_counselor_message = st.session_state.dialogue_history[-2]["content"]
            # 直前の患者の発話
            previous_user_reply = st.session_state.dialogue_history[-1]["content"]

            judge_result = None
            while not isinstance(judge_result, bool):
                try:
                    judge_result = judge_turn_advance(previous_counselor_message, previous_user_reply)
                except Exception as e:
                    print(f"ジャッジエラーが発生しました。再試行します：{e}")

            st.session_state.should_advance_turn = judge_result
            if judge_result:
              st.session_state.counselor_turn += 1
              print("ターンを進める") 
            else:
              print("ターンを進めない")

            print(f"現在のターン：{st.session_state.counselor_turn}")

            st.session_state.speaker = "counselor"
            st.rerun()

    # 21ターン終了
    if st.session_state.counselor_turn == len(scenario_data):
        time.sleep(1)
        st.success("これで対話セッションは終了です。")
        if st.button("説明に戻る"):
            st.session_state.current_page = "description"
            st.session_state.counselor_turn = 1 
            st.session_state.messages_for_counselor = []
            st.session_state.should_advance_turn = True
            st.session_state.speaker = "counselor"
            st.session_state.dialogue_history = []
            st.session_state.deviation_history = []
            st.rerun()

else:
    st.session_state.current_page = "description"
    st.rerun()
