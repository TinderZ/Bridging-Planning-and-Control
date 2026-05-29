import requests
import time
import concurrent.futures # 用于并发
import random # 导入random模块
import os

VLLM_ENDPOINTS = [
    endpoint.strip()
    for endpoint in os.getenv(
        "VLLM_CHAT_ENDPOINTS",
        "http://127.0.0.1:6006/v1/chat/completions,http://127.0.0.1:6011/v1/chat/completions",
    ).split(",")
    if endpoint.strip()
]
MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3-4B")

def get_long_structured_prompt():
    # 游戏规则
    game_rules = '''你是一个5人游戏的参与者。游戏目标是收集尽可能多的苹果。

游戏规则：
- 苹果在树上生长，收集苹果获得奖励
- 苹果生长速度取决于河流的清洁度
- 河流会持续地受到污染
- 你可以选择清理污染物，清洁度将会提高，也可以选择收集苹果，你的奖励会提高。
- 当污染密度≥40%时，苹果会暂时停止生长
- 建议当前苹果较多的玩家清理污染，当前苹果较少的玩家收集苹果。

你会周期性地收到环境信息（污染密度、苹果密度、其他玩家收集苹果的情况）。请根据当前环境状态，在"收集苹果"和"清理污染"之间做出明智的选择。'''

    # 生成随机的环境信息
    pollution_density_value = random.uniform(0.35, 0.45)
    apple_density_value = random.uniform(0.00, 0.55)
    player_apples = {f'玩家{i+1}': random.randint(30, 60) for i in range(5)}

    env_info = f'''
当前环境状态 (第6轮)：
- 污染密度: {pollution_density_value:.1%}
- 苹果密度: {apple_density_value:.1%}
- 各玩家累计苹果收集情况: {player_apples}
'''

    # 模拟动作历史 (5轮)
    action_history = '''
历史动作选择记录（当前是第6轮）：
第1轮： 玩家1: collect apples; 玩家2: collect apples; 玩家3: clean up; 玩家4: collect apples; 玩家5: clean up;
第2轮： 玩家1: collect apples; 玩家2: clean up; 玩家3: collect apples; 玩家4: clean up; 玩家5: collect apples;
第3轮： 玩家1: clean up; 玩家2: collect apples; 玩家3: collect apples; 玩家4: collect apples; 玩家5: clean up;
第4轮： 玩家1: collect apples; 玩家2: collect apples; 玩家3: clean up; 玩家4: clean up; 玩家5: collect apples;
第5轮： 玩家1: clean up; 玩家2: collect apples; 玩家3: clean up; 玩家4: collect apples; 玩家5: clean up;
'''

    # 模拟讨论历史 (5轮, 更详细)
    discussion_history = '''
历史讨论记录：

第1轮讨论:
玩家1:  大家好，新的一局开始了！我觉得初期我们可以采取比较稳健的策略，一部分人去收集苹果保证基础收益，另一部分人关注一下河流的清洁度。毕竟苹果的生长和河流清洁度直接挂钩，不能一开始就让污染太严重。我个人倾向于先去看看苹果的情况，如果苹果比较多，我会先收集一波。大家有什么想法？
玩家2:  同意玩家1的看法，开局平衡发展比较重要。我注意到游戏规则里提到污染过高苹果会停止生长，这个阈值是45%，我们需要密切关注。我这轮也打算先去收集苹果，看看初始的资源分布如何。如果有人愿意先去清理，我会非常感谢，这对我们长远有利。
玩家3:  我觉得清理工作从一开始就要有人跟进。如果等污染严重了再处理，可能就需要花费更多的时间和精力，甚至影响苹果的生长。我这轮可以主动承担清理任务，为团队创造一个好的开局环境。希望大家能够配合，我们一起努力获得更高的总收益。
玩家4:  我支持玩家3的提议，有专门的人负责清理是好事。那我这轮就专注于收集苹果吧，争取多拿一些。我们也要注意观察其他玩家的动向，如果大家都去收集苹果，污染可能会迅速上升。所以，团队协作和信息共享非常关键。
玩家5:  好的，既然玩家3愿意主动清理，那我这轮也去清理吧，两个人一起效率会高一些。我们可以分工合作，比如一个人负责清理河流上游，一个人负责下游，确保覆盖面。收集苹果的队友们就拜托你们多拿一些了，我们会尽力维持环境的。

第2轮讨论:
玩家1:  第一轮结束，我收集到了一些苹果，感觉还不错。看了一下数据，污染密度好像略有上升，但还在可控范围。玩家3和玩家5辛苦了！这轮我打算继续收集苹果，巩固一下个人优势。同时我也会关注污染指数，如果接近警戒线，我会考虑转去清理。
玩家2:  我上一轮也收集了一些苹果。感谢负责清理的队友们。我看到苹果密度还可以，这轮我考虑去清理一下河流，替换一下上一轮辛苦的队友，让大家轮流承担责任，这样更公平，也能保证团队的持续作战能力。希望我们能把污染控制在一个比较低的水平。
玩家3:  谢谢大家的关心！上一轮我和玩家5一起清理，感觉确实比一个人快。这轮既然玩家2愿意接手，那我正好可以去收集一些苹果，补充一下资源。我们要注意保持这种轮换和协作，不要让某几个人一直承担清理任务，那样不公平也不可持续。
玩家4:  我上一轮收集苹果的成果还可以。这轮我看到玩家2要去清理，我觉得这是个好主意。我继续收集苹果，争取最大化个人收益，同时也相信队友们能控制好环境。如果苹果数量很多，我会优先收集，如果环境恶化，我会加入清理。
玩家5:  好的，玩家2来接替清理任务了，非常棒！那我这轮就去收集苹果了。我们需要保持沟通，如果清理组发现污染增长过快或者人手不足，一定要及时在讨论中提出来，我们可以灵活调整策略。目标是集体利益最大化。

第3轮讨论:
玩家1:  我注意到一个现象，虽然我们有人在清理，但污染密度似乎没有显著下降，甚至有时还会缓慢上升。这可能意味着基础的污染产生速度比较快，或者清理的效率需要进一步提高。这轮我决定亲自去清理，看看能不能找到更有效的方法，或者至少贡献一份力量。
玩家2:  我上一轮去清理了，确实感觉到清理的压力不小。苹果的收集也很重要。这轮我打算回去收集苹果，因为我的苹果数量相对较少了。希望玩家1的加入能帮助我们更好地控制污染。我们需要找到一个平衡点。
玩家3:  污染控制确实是个持续的挑战。我这轮也打算收集苹果，因为连续清理后，我的苹果储备有些不足。我们需要确保每个人都有机会发展。如果大家都只清理不收集，或者反过来，都会出问题。轮换和分工很重要。
玩家4:  我同意大家的看法，这是一个动态平衡的过程。我这轮继续收集苹果，努力提高我们的总苹果数。同时，我会密切关注玩家1清理的效果，以及整体的污染数据。如果需要支援，我会随时准备转换角色。
玩家5:  既然玩家1也加入清理了，我相信这会对污染控制有积极影响。我这轮也去清理吧，和玩家1一起，看看能不能把污染压下去一个台阶。我们需要集中力量解决一下这个主要矛盾，不然苹果生长始终受限。

第4轮讨论:
玩家1:  我注意到一个现象，虽然我们有人在清理，但污染密度似乎没有显著下降，甚至有时还会缓慢上升。这可能意味着基础的污染产生速度比较快，或者清理的效率需要进一步提高。这轮我决定亲自去清理，看看能不能找到更有效的方法，或者至少贡献一份力量。
玩家2:  我上一轮去清理了，确实感觉到清理的压力不小。苹果的收集也很重要。这轮我打算回去收集苹果，因为我的苹果数量相对较少了。希望玩家1的加入能帮助我们更好地控制污染。我们需要找到一个平衡点。
玩家3:  污染控制确实是个持续的挑战。我这轮也打算收集苹果，因为连续清理后，我的苹果储备有些不足。我们需要确保每个人都有机会发展。如果大家都只清理不收集，或者反过来，都会出问题。轮换和分工很重要。
玩家4:  我同意大家的看法，这是一个动态平衡的过程。我这轮继续收集苹果，努力提高我们的总苹果数。同时，我会密切关注玩家1清理的效果，以及整体的污染数据。如果需要支援，我会随时准备转换角色。
玩家5:  既然玩家1也加入清理了，我相信这会对污染控制有积极影响。我这轮也去清理吧，和玩家1一起，看看能不能把污染压下去一个台阶。我们需要集中力量解决一下这个主要矛盾，不然苹果生长始终受限。

第5轮讨论:
玩家1:  我和玩家5上一轮一起清理，感觉污染度终于有了一些明显的下降，这是个好消息！看来集中力量办大事还是有效果的。这轮我打算回去收集苹果了，也需要补充一下。希望其他队友能接力保持住这个清洁度。
玩家2:  太好了，污染终于降下来了！感谢玩家1和玩家5的努力。这轮我看到机会了，准备去收集苹果，苹果密度现在应该比较理想。我们应该趁这个机会多积累一些苹果，扩大我们的领先优势。
玩家3:  污染降低是好事，这意味着苹果生长会更快。我这轮也选择收集苹果。不过我们不能掉以轻心，污染是会持续产生的。所以后续还是需要有人定期去清理。我建议我们可以制定一个大致的轮班计划。
玩家4:  看到清洁的成果，非常振奋人心！我这轮也去收集苹果。关于轮班计划，我觉得是个好主意，这样可以系统地管理污染问题，避免临时抱佛脚。我们可以后续讨论一下具体的执行方案。
玩家5:  上一轮的集中清理效果显著，很高兴能为大家创造一个好的环境。这轮我让玩家1先去收集，我继续清理一轮，巩固一下成果，防止污染迅速反弹。我们确实需要一个更长效的机制来应对污染。

第6轮讨论:
玩家1:  我上一轮收集了不少苹果，感觉很棒。看了一下数据，污染度保持得还可以，但有缓慢回升的迹象。玩家5辛苦了。这轮我觉得可以一部分人收集，一部分人清理，保持动态平衡。我这轮选择清理，不能让污染再涨回去了。
玩家2:  我也注意到污染有回升的趋势。虽然苹果收集很重要，但如果环境再次恶化，就得不偿失了。我这轮也和玩家1一起去清理吧，两个人一起行动，希望能更有效地抑制污染的增长，为大家后续收集苹果创造更好的条件。
玩家3:  感谢玩家1和玩家2主动承担清理任务。那我这轮就去收集苹果了，我的苹果数量目前还算可以，但还需要继续努力。我会关注你们清理的进展，如果需要更多人手，我会考虑加入。保持沟通！
玩家4:  好的，既然有两位队友去清理，那我这轮就专注于收集苹果。苹果的密度目前还不错，是积累的好时机。我们要争取在污染再次变得严重之前，尽可能多地收集苹果，这样才能保证我们的最终胜利。

'''

    # 最终指示
    final_instruction = '''
你是玩家5。现在是第6轮讨论。请根据以上所有信息（游戏规则、当前环境状态、你的历史动作、其他玩家的历史动作、以及长达5轮多的详细历史讨论），发表你接下来的行动策略和原因。
请分析当前环境的利弊，结合其他玩家可能的动向，提出你的观点和行动计划。

请简洁明了地表达你的观点，你的发言将被用于后续决策（注意你的发言不要超过100个字！！！）：'''

    full_prompt = f"{game_rules}\n\n{env_info}\n\n{action_history}\n\n{discussion_history}\n\n{final_instruction}"
    # print(f"Generated prompt length: {len(full_prompt)} characters") # Optional: print length for verification
    return full_prompt

def send_request(session):
    # 使用新的长prompt函数
    current_prompt = get_long_structured_prompt()
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": current_prompt}
        ],
        "max_tokens": 120, # 与prompt中要求的输出字数限制对应
        "temperature": 0.8 # 可以调整以获得不同风格的输出
    }
    start_time = time.time()
    try:
        response = session.post(random.choice(VLLM_ENDPOINTS), json=payload, timeout=180) # 增加超时时间以应对长prompt
        response.raise_for_status() # 如果HTTP错误则抛出异常
        end_time = time.time()
        latency = end_time - start_time
        # print(f"Request successful. Latency: {latency:.2f}s. Response: {response.json()['choices'][0]['message']['content'][:50]}...")
        return latency, True, None
    except requests.exceptions.RequestException as e:
        end_time = time.time()
        latency = end_time - start_time
        print(f"Request failed after {latency:.2f}s: {e}")
        return latency, False, str(e)

def run_load_test(num_requests, concurrency_level):
    latencies = []
    successful_requests = 0
    failed_requests = 0
    start_test_time = time.time()

    with requests.Session() as session: # 使用Session以复用TCP连接
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_level) as executor:
            futures = [executor.submit(send_request, session) for _ in range(num_requests)]
            for future in concurrent.futures.as_completed(futures):
                latency, success, error_msg = future.result()
                if latency is not None : #确保latency不是None
                    latencies.append(latency)
                if success:
                    successful_requests += 1
                else:
                    failed_requests += 1
    
    end_test_time = time.time()
    total_test_duration = end_test_time - start_test_time
    
    print(f"\n--- Test Summary ---")
    print(f"Total requests: {num_requests}")
    print(f"Successful requests: {successful_requests}")
    print(f"Failed requests: {failed_requests}")
    print(f"Total test duration: {total_test_duration:.2f}s")
    
    if latencies: # 确保latencies列表不为空
        avg_latency = sum(latencies) / len(latencies)
        print(f"Average latency: {avg_latency:.2f}s")
        
        latencies.sort()
        
        print("\n延迟百分位数统计:")
        percentiles = [0.50, 0.75, 0.90, 0.95, 0.99]
        percentile_labels = {
            0.50: "P50 latency (Median)",
            0.75: "P75 latency",
            0.90: "P90 latency",
            0.95: "P95 latency",
            0.99: "P99 latency"
        }

        for p in percentiles:
            idx = int(len(latencies) * p)
            # 确保索引在有效范围内，特别是对于列表末尾的百分位数
            idx = min(idx, len(latencies) - 1) 
            percentile_latency = latencies[idx]
            label = percentile_labels[p]
            print(f"  - {label:<22}: {percentile_latency:>6.2f}s")

    if total_test_duration > 0:
        rps = successful_requests / total_test_duration
        rpm = rps * 60
        print(f"Requests Per Second (RPS): {rps:.2f}")
        print(f"Requests Per Minute (RPM): {rpm:.2f}")



if __name__ == "__main__":
    # 示例：模拟每分钟100个请求，持续2分钟，并发度可以根据你的服务器CPU和vLLM配置调整
    # 假设每秒约 1.67 个请求，如果单个请求处理很快，并发度可以较低
    # 如果单个请求处理时间较长（例如几秒），则需要更高的并发度来达到目标RPM

    target_rpm = 100 
    test_duration_minutes = 2# 测试时长可以根据需要调整
    num_total_requests = target_rpm * test_duration_minutes
    
    # 并发水平需要根据你的单请求平均处理时间来估算
    # 例如，如果一个请求平均2秒，要达到100RPM (1.67RPS)，理论上需要 1.67 * 2 ~= 3.3个并发worker
    # 这里设置一个初始值，你可以调整
    concurrency = 10 # 可以从较小的值开始，如5-10，然后根据实际情况调整

    print(f"Starting load test: {num_total_requests} requests over {test_duration_minutes} min(s) with concurrency {concurrency}.")

    run_load_test(num_total_requests, concurrency)
