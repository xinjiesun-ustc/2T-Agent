import json
from config import DATA_PATH, DATA_PATH_TATA

class UserProfile:
    """
    Learner Ability Profile Module
    只处理个人能力与平均能力的判断
    """
    def __init__(self, agent_id):
        self.agent_id = str(agent_id)
        profile_path = f"{DATA_PATH_TATA}/C697360_user_profile.json"
        # profile_path = f"{DATA_PATH}/profile.json"
        with open(profile_path, encoding='utf-8') as f:
            self.theta_data = json.load(f)

    def ability(self):
        # 格式为 "5\t0.58000624\t0.4936993481999999\n"
        theta_line = self.theta_data[self.agent_id]
        _, ability, avg = theta_line.strip().split('\t')
        ab = float(ability)
        avg = float(avg)

        if ab > 0.5 and ab > 1.2 * avg:
            return 'good'
        elif ab > avg:
            return 'common'
        else:
            return 'poor'

    def build_User_prompt(self):    #用于大模型所扮演的角色
        return (
            f"You are an intelligent agent who thinks like a teacher, capable of evaluating students' learning performance and assessing the difficulty of exercises.\n"
            f"You believe the student's current problem-solving ability is {self.ability()},"
        )

# class ExerciseProfile:
#     """
#     Exercise Difficulty Profile Module
#     只处理题目的难度与平均难度的判断
#     """
#     def __init__(self, item_id):
#         self.item_id = str(item_id)
#         profile_path = f"{DATA_PATH_TATA}/C697360_exercise_profile.json"
#         with open(profile_path, encoding='utf-8') as f:
#             self.difficulty_data = json.load(f)
#
#     def difficulty(self):
#         # 格式为 "1\t0.95396334\t0.527453745\n"
#         line = self.difficulty_data[self.item_id]
#         _, diff, avg = line.strip().split('\t')
#         d = float(diff)
#         avg = float(avg)
#
#         if d > 0.5 and d > 1.2 * avg:
#             return 'Very Difficult'
#         elif d > avg:
#             return 'Difficult'
#         else:
#             return 'Easy'
#     def build_Exercise_prompt(self):    #用于大模型所扮演的角色
#
#         return (
#             "You are an intelligent agent who thinks like a teacher and can assess the difficulty of exercises.\n"
#             f"I believe the difficulty of this problem is {self.difficulty()}."
#         )




def get_exercise_difficulty(item_id):
    """
    根据题目 item_id 返回其难度评级：'Very Difficult', 'Difficult', 或 'Easy'
    参数:
        item_id: str 或 int，题目编号
        data_path: str，路径，指向 C697360_exercise_profile.json 文件
    返回:
        难度等级字符串
    """
    item_id = str(item_id)
    profile_path = f"{DATA_PATH_TATA}/C697360_exercise_profile.json"

    with open(profile_path, encoding='utf-8') as f:
        difficulty_data = json.load(f)

    line = difficulty_data[item_id]
    _, diff, avg = line.strip().split('\t')
    d = float(diff)
    avg = float(avg)

    if d > 0.5 and d > 1.2 * avg:
        return 'Very Difficult'
    elif d > avg:
        return 'Difficult'
    else:
        return 'Easy'


def build_exercise_prompt(item_id):
    """
    构建传给大模型的提示词 prompt
    参数:
        item_id: 题目编号
        data_path: 数据路径
    返回:
        prompt 字符串
    """
    difficulty = get_exercise_difficulty(item_id)
    return (
        f"and that the difficulty of this problem is {difficulty}."
    )


if __name__ == '__main__':
    profile = UserProfile(16)
    print("Ability Level:", profile.ability())
    print(profile.build_User_prompt())

    exercise = ExerciseProfile(1)
    print("Difficulty Level:", exercise.difficulty())
    print(exercise.build_Exercise_prompt())





