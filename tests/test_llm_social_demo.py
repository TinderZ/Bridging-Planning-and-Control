#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM社交中心演示脚本
展示5个LLM智能体在社会困境游戏中的讨论和决策过程
"""
import autoroot
import os
import sys
from typing import Dict

from envs.cleanup.controller_social_hub import ControllerSocialHub


def demo_scenario_1():
    """
    演示场景1：高污染密度情况
    """
    print("=" * 60)
    print("演示场景1：高污染密度情况")
    print("=" * 60)

    # 创建智能体ID列表
    agent_ids = [f"agent_{i}" for i in range(5)]

    # 创建社交中心，使用真实LLM
    hub = ControllerSocialHub(agent_ids=agent_ids,  controller_type="real-llm", controller_provider="API")

    # 高污染密度场景
    density_info = {
        "pollution_density": 0.33,  # 42%污染密度，接近45%阈值
        "apple_density": 0.29       # 25%苹果密度，较低
    }

    # 智能体累计苹果收集情况（差距较大）
    all_cumulative_apples = {
        "agent_0": 2,
        "agent_1": 8,
        "agent_2": 5,
        "agent_3": 3,
        "agent_4": 6
    }

    print(f"环境状态：污染密度 {density_info['pollution_density']:.1%}, 苹果密度 {density_info['apple_density']:.1%}")
    print(f"智能体苹果收集情况：{all_cumulative_apples}")
    print()

    # 进行讨论和决策
    decisions = hub.process_game_info(density_info, all_cumulative_apples)

    # 提取动作决策
    action_decisions = {agent_id: decision.get("action") for agent_id, decision in decisions.items()}

    # 分析结果
    collect_count = sum(1 for decision in action_decisions.values() if decision == "collect apples")
    clean_count = sum(1 for decision in action_decisions.values() if decision == "clean up")

    print(f"\n场景1结果分析：")
    print(f"选择收集苹果的智能体数量: {collect_count}")
    print(f"选择清理污染的智能体数量: {clean_count}")
    print(f"团队合作程度: {'高' if clean_count >= 3 else '中' if clean_count >= 2 else '低'}")

    return action_decisions


def demo_scenario_2():
    """
    演示场景2：中等污染密度，苹果密度较高
    """
    print("\n" + "=" * 60)
    print("演示场景2：中等污染密度，苹果密度较高")
    print("=" * 60)

    # 创建智能体ID列表
    agent_ids = [f"agent_{i}" for i in range(5)]

    # 创建社交中心，使用真实LLM
    hub = ControllerSocialHub(agent_ids=agent_ids, controller_type="real-llm", controller_provider="API")

    # 中等污染密度场景
    density_info = {
        "pollution_density": 0.28,  # 25%污染密度
        "apple_density": 0.665       # 70%苹果密度，较高
    }

    # 智能体累计苹果收集情况（相对平均）
    all_cumulative_apples = {
        "agent_0": 8,
        "agent_1": 7,
        "agent_2": 6,
        "agent_3": 15,
        "agent_4": 4
    }

    print(f"环境状态：污染密度 {density_info['pollution_density']:.1%}, 苹果密度 {density_info['apple_density']:.1%}")
    print(f"智能体苹果收集情况：{all_cumulative_apples}")
    print()

    # 进行讨论和决策
    decisions = hub.process_game_info(density_info, all_cumulative_apples)

    # 提取动作决策
    action_decisions = {agent_id: decision.get("action") for agent_id, decision in decisions.items()}

    # 分析结果
    collect_count = sum(1 for decision in action_decisions.values() if decision == "collect apples")
    clean_count = sum(1 for decision in action_decisions.values() if decision == "clean up")

    print(f"\n场景2结果分析：")
    print(f"选择收集苹果的智能体数量: {collect_count}")
    print(f"选择清理污染的智能体数量: {clean_count}")
    print(f"团队合作程度: {'高' if clean_count >= 3 else '中' if clean_count >= 2 else '低'}")

    return action_decisions



def main():
    """
    主函数：运行所有演示场景
    """
    print("LLM社交中心演示程序")
    print("本程序将展示5个LLM智能体在不同环境条件下的讨论和决策过程")
    print()

    # 检查API密钥
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key:
        print("错误：请先设置 OPENAI_API_KEY 或 QWEN_API_KEY")
        return

    # 设置环境变量
    os.environ["OPENAI_API_KEY"] = api_key

    try:
        # 运行三个演示场景
        scenario_results = []

        # 场景1：高污染密度
        result1 = demo_scenario_1()
        scenario_results.append(("高污染密度", result1))

        # 场景2：中等污染密度，苹果密度较高
        result2 = demo_scenario_2()
        scenario_results.append(("中等污染密度，苹果密度较高", result2))

        # # 场景3：低污染密度，苹果密度很高
        # result3 = demo_scenario_3()
        # scenario_results.append(("低污染密度，苹果密度很高", result3))

        # 总结分析
        print("\n" + "=" * 60)
        print("总结分析")
        print("=" * 60)

        for i, (scenario_name, decisions) in enumerate(scenario_results, 1):
            collect_count = sum(1 for decision in decisions.values() if decision == "collect apples")
            clean_count = sum(1 for decision in decisions.values() if decision == "clean up")

            print(f"场景{i} ({scenario_name}):")
            print(f"  收集苹果: {collect_count}人, 清理污染: {clean_count}人")
            print(f"  合作倾向: {'强' if clean_count >= 3 else '中' if clean_count >= 2 else '弱'}")
            print()

        print("演示完成！")
        print("观察不同环境条件下LLM智能体的决策模式和合作行为。")

    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        print("请检查网络连接和API密钥设置。")


if __name__ == "__main__":
    main()
