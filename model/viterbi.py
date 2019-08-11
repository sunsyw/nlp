# 状态的样本空间
states = ('健康', '发热')
# 观测的样本空间
observations = ('正常', '发冷', '发晕')
# 起始个状态概率
start_probability = {'健康': 0.6, '发热': 0.4}
# 状态转移概率
transition_probability = {
    '健康': {'健康': 0.7, '发热': 0.3},
    '发热': {'健康': 0.4, '发热': 0.6},
}
# 状态->观测的发散概率
emission_probability = {
    '健康': {'正常': 0.5, '发冷': 0.4, '发晕': 0.1},
    '发热': {'正常': 0.1, '发冷': 0.3, '发晕': 0.6},
}


def viterbi(obs, states, start_p, trans_p, emit_p):
    result_m = [{}]
    # 存放结果,每一个元素是一个字典，每一个字典的形式是 state:(p,pre_state)
    # 其中state,p分别是当前状态下的概率值，pre_state表示该值由上一次的那个状态计算得到
    for s in states:  # 对于每一个状态 ('健康', '发热')
        result_m[0][s] = (start_p[s] * emit_p[s][obs[0]], None)
        # 把第一个观测节点对应的各状态值计算出来
        # 健康 = 0.6 * 0.5 = 0.3
        # 发热 = 0.4 * 0.1 = 0.04
        # [{健康: (0.3, None), 发热: (0.04, None)}]

    for t in range(1, len(obs)):  # 1, 2
        result_m.append({})
        # 准备t时刻的结果存放字典，形式同上

        for s in states:  # ('健康', '发热')
            # 对于每一个t时刻状态s,获取t-1时刻每个状态s0的p,结合由s0转化为s的转移概率和s状态至obs的发散概率
            # 计算t时刻s状态的最大概率，并记录该概率的来源状态s0
            # max()内部比较的是一个tuple:(p,s0),max比较tuple内的第一个元素值
            result_m[t][s] = max([(result_m[t - 1][s0][0] * trans_p[s0][s] * emit_p[s][obs[t]], s0) for s0 in states])
            # 发冷: 健康: 健康: 0.3 * 0.7 * 0.4 = 0.084   发热: 0.04 * 0.4 * 0.4 = 0.0064   -> 0.084
            #      发热: 健康: 0.3 * 0.3 * 0.3 = 0.027   发热: 0.04 * 0.6 * 0.3 = 0.0072   -> 0.027
            # 发晕: 健康: 健康: 0.084 * 0.7 * 0.1 = 0.00588   发热: 0.027 * 0.4 * 0.1 = 0.00108   -> 0.00588
            #      发热: 健康: 0.084 * 0.3 * 0.6 = 0.01512   发热: 0.027 * 0.6 * 0.6 = 0.00972   -> 0.01512
            # [{健康: (0.3, None), 发热: (0.04, None)},
            # {健康: (0.084, 健康), 发热: (0.027, 健康)},
            # {健康: (0.00588, 健康), 发热: (0.01512, 健康)}]
    return result_m
    # 所有结果（包括最佳路径）都在这里，但直观的最佳路径还需要依此结果单独生成，在显示的时候生成


result_m = viterbi(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability)
print(result_m)

# import math
#
# # 状态的样本空间
# states = ('健康', '发热')
# # 观测的样本空间
# observations = ('正常', '发冷', '发晕')
# # 起始个状态概率
# start_probability = {'健康': 0.6, '发热': 0.4}
# # 状态转移概率
# transition_probability = {
#     '健康': {'健康': 0.7, '发热': 0.3},
#     '发热': {'健康': 0.4, '发热': 0.6},
# }
# # 状态->观测的发散概率
# emission_probability = {
#     '健康': {'正常': 0.5, '发冷': 0.4, '发晕': 0.1},
#     '发热': {'正常': 0.1, '发冷': 0.3, '发晕': 0.6},
# }
#
#
# # 计算以E为底的幂
# def E(x):
#     # return math.pow(math.e,x)
#     return x
#
#
# def display_result(observations, result_m):
#     """
#     较为友好清晰的显示结果
#     :param result_m:
#     :return:
#     """
#     # 从结果中找出最佳路径
#     infered_states = []
#     final = len(result_m) - 1
#     (p, pre_state), final_state = max(zip(result_m[final].values(), result_m[final].keys()))
#     infered_states.insert(0, final_state)
#     infered_states.insert(0, pre_state)
#     for t in range(final - 1, 0, -1):
#         _, pre_state = result_m[t][pre_state]
#         infered_states.insert(0, pre_state)
#     print(format("Viterbi Result", "=^59s"))
#     head = format("obs", " ^10s")
#     head += format("Infered state", " ^18s")
#     for s in states:
#         head += format(s, " ^15s")
#     print(head)
#     print(format("", "-^59s"))
#
#     for obs, result, infered_state in zip(observations, result_m, infered_states):
#         item = format(obs, " ^10s")
#         item += format(infered_state, " ^18s")
#         for s in states:
#             item += format(result[s][0], " >12.8f")
#             if infered_state == s:
#                 item += "(*)"
#             else:
#                 item += "   "
#
#         print(item)
#     print(format("", "=^59s"))
#
#
# def viterbi(obs, states, start_p, trans_p, emit_p):
#     result_m = [{}]  # 存放结果,每一个元素是一个字典，每一个字典的形式是 state:(p,pre_state)
#     # 其中state,p分别是当前状态下的概率值，pre_state表示该值由上一次的那个状态计算得到
#     for s in states:  # 对于每一个状态
#         result_m[0][s] = (E(start_p[s] * emit_p[s][obs[0]]), None)  # 把第一个观测节点对应的各状态值计算出来
#
#     for t in range(1, len(obs)):
#         result_m.append({})  # 准备t时刻的结果存放字典，形式同上
#
#         for s in states:  # 对于每一个t时刻状态s,获取t-1时刻每个状态s0的p,结合由s0转化为s的转移概率和s状态至obs的发散概率
#             # 计算t时刻s状态的最大概率，并记录该概率的来源状态s0
#             # max()内部比较的是一个tuple:(p,s0),max比较tuple内的第一个元素值
#             result_m[t][s] = max(
#                 [(E(result_m[t - 1][s0][0] * trans_p[s0][s] * emit_p[s][obs[t]]), s0) for s0 in states])
#     return result_m  # 所有结果（包括最佳路径）都在这里，但直观的最佳路径还需要依此结果单独生成，在显示的时候生成
#
#
# def example():
#     """
#     一个可以交互的示例
#     """
#     result_m = viterbi(observations,
#                        states,
#                        start_probability,
#                        transition_probability,
#                        emission_probability)
#     display_result(observations, result_m)
#     while True:
#         user_obs = input("轮到你来输入观测,计算机来推断可能状态\n"
#                          "使用 'N' 代表'正常', 'C' 代表'发冷','D'代表'发晕'\n"
#                          "您输入：('q'将退出):")
#
#         if len(user_obs) == 0 or 'q' in user_obs or 'Q' in user_obs:
#             break
#         else:
#             obs = []
#             for o in user_obs:
#                 if o.upper() == 'N':
#                     obs.append("正常")
#                 elif o.upper() == 'C':
#                     obs.append("发冷")
#                 elif o.upper() == 'D':
#                     obs.append("发晕")
#                 else:
#                     pass
#             result_m = viterbi(obs,
#                                states,
#                                start_probability,
#                                transition_probability,
#                                emission_probability)
#             display_result(obs, result_m)
#
#
# if __name__ == "__main__":
#     example()
