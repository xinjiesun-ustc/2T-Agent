import asyncio
from config import DATA_PATH, RESULT_PATH
from Myprofile import  UserProfile
from memory import Memory
from action import AgentAction
from utils import load_json, save_json
import os
import  json
####下面的是对整个数据集 适合冷启动
# async def run_for_student(student_index):  #异步函数，用于模拟单个学生
#     print(student_index)
#     all_logs = load_json(f"{DATA_PATH}/C_697360_converted_user_records_list_final.json")
#     logs = all_logs[student_index]['logs']
#     student_id = all_logs[student_index]['user_id']
#
#
# ###作为教师Agent，我们同时考虑用户画像和题目画像，代表老师对学生和题目的掌握情况，符合老师的特质
#     User_profile = UserProfile(student_id)  #用户画像
#     # Exercise_profile = ExerciseProfile(exercise_id) #练习题目画像在Myprofile中的build_exercise_prompt考虑
#
#     memory = Memory()
#     action = AgentAction(User_profile, memory)
#     results = []
#
#     for rec_id in range(len(logs)):
#         rec = logs[rec_id]  #当前学生的此刻正在处理的练习记录
#         ans, raw, corr, summ = action.simulate_step(rec)
#         results.append({'ans':ans,'raw':raw,'corr':corr,'summ':summ})
#
#     save_json(f"{RESULT_PATH}/{student_id}_results.json", results)
####下面的是对测试集进行正确率统计的
async def run_for_student(student_index):  #异步函数，用于模拟单个学生
    print(f"📘 处理学生：{student_index}")
    all_logs = load_json(f"{DATA_PATH}/C_697360_converted_user_records_list_final.json")
    logs = all_logs[student_index]['logs']
    student_id = all_logs[student_index]['user_id']

    # 分割训练/测试
    split_point = int(0.7 * len(logs))  # 你可以调整比例
    train_logs = logs[:split_point]
    test_logs = logs[split_point:]

    User_profile = UserProfile(student_id)  # 用户画像
    memory = Memory()
    action = AgentAction(User_profile, memory)

    # ▶️ 训练阶段
    for rec in train_logs:
        _ = action.simulate_step(rec)  # 只学习，结果不记录

    # 📊 测试阶段
    # 📊 测试阶段：计算 task2 的预测准确率
    results = []
    correct_count = 0

    for rec in test_logs:
        ans, raw, corr, summ = action.simulate_step(rec)

        # Task 2 预测逻辑（Yes -> 1, No -> 0）
        task2_pred = 1 if ans.get('task2', '').strip().lower() == 'yes' else 0
        actual_score = rec['is_correct']  # 实际标签

        is_correct = int(task2_pred == actual_score)
        correct_count += is_correct

        results.append({
            'ans': ans,
            'raw': raw,
            'corr': is_correct,  # 注意：这里记录的是 task2 的正确与否
            'summ': summ
        })

    accuracy = correct_count / len(test_logs) if test_logs else 0.0
    print(f"🎯 Task 2 Prediction Accuracy for student {student_id}: {accuracy:.2%}")
    # ✅ 保存结果到文件（以追加方式写入）
    result = {
        "student_id": student_id,
        "accuracy": accuracy
    }

    with open("C_697360_accuracy_results.jsonl", "a", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
        f.write("\n")


async def main():
    students = range(1)
    # user_idx_start = 0
    # user_idx_end = 1
    # agent_id_list = load_json(f"{DATA_PATH}/agent_id_list.json")
    # students = agent_id_list[user_idx_start:user_idx_end]
    
    #await asyncio.gather(*[run_for_student(s) for s in students])   #并发地对多个学生执行 run_for_student() 函数。当前设置只跑一个学生（range(1)
    # students = range(5)
    # for s in students:
    #     await run_for_student(s)
    await run_for_student(4)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    asyncio.run(main())
