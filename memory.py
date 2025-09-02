import math, random
from config import SIM_PARAMS

class Memory:
    """Memory Module：Factual/Short/Long + 检索/写入/反思"""

    def __init__(self):
        self.factual = []
        self.short = []
        self.long = {

            'learning_status': [],  ###用于存放LLM关于reflect_summary的反思结果，每一步都把反思结果存放起来
            'practiced_knowledge': []
        }
        self.short_size = SIM_PARAMS['short_term_size']
        # self.knowledgeconcept_name = knowledgeconcept_name

    def write_factual(self, record):  #老师了解学生做题表现
        # 写入新的观察
        self.factual.append(record)

    def write_long_summary_and_kc(self, summary,KC_name):
        self.long['learning_status'].append(summary)
        self.long['practiced_knowledge'].append(KC_name)

    def retrieve_short(self): #老师了解学生的最近答题表现
        # 返回(检索)最近 s 条
        self.short = self.factual[-self.short_size:]
        return self.short

    def retrieve_long(self):   ###每次都返回以前所有题目的反思和知识点，感觉会把Prompt变得非常长啊，想一个解决办法？
        return {
            'learning_status': self.long['learning_status'],  ###系统或LLM生成的反思性总结的prompt
            'practiced_knowledge': self.long['practiced_knowledge']
        }


    def reflect_corrective(self, practice, ans):
        fb = ''
        ##反思有没有把最终答案预测正确
        if (ans.get('task2', '').lower() != 'yes') and practice['score'] == 1:
            fb += "You thought you couldn't solve this problem correctly, but in fact, you will solve it correctly. \n"
        if (ans.get('task2', '').lower() != 'no') and practice['score'] == 0:
            fb += "You thought you could solve this problem correctly, but in fact, you does not answer it correctly. \n"
        return fb

    def reflect_summary(self, corrective):  ##corrective就是纠正性反思的fb   构建一个基于前面反思的提示语（作为 prompt），但本函数本身没有重新生成总结内容，仅作为提示加入，除非前面没有生成反思的情况下进行新的反思总结
        if corrective != '':
            summary = corrective
        else:
            summary = ''
        summary += f"\n You should directly output your reflection and summarize your # Learning Status # within 500 words based on your # profile #, # short-term memory #, # long-term memory # and previous # Learning Status #. Do not output any other information."

        # self.long['learning_status'].append(summary)  #感觉这里没必要要这个啊 后面action会把真正的LLM返回的summary写入 self.long
        return summary

def main():
    import random



    # 知识名称标准化映射
    know_name = {
        "loop": "loop",
        "for loop": "for loop",
        "if statement": "if statement",
        "photosynthesis": "photosynthesis"
    }

    # 初始化 Memory 对象
    memory = Memory(knowledgeconcept_name=know_name)

    # 模拟写入 factual memory 的知识点记录（格式：[id, knowledge_name, timestamp, count]）
    memory.write_factual(["stu1", "loop", 1, 1])
    memory.write_factual(["stu1", "for loop", 2, 1])
    memory.write_factual(["stu1", "photosynthesis", 3, 1])

    # 强化一个新知识点，与现有某些相关
    new_record = ["stu1", "for loop", 4]  # 将要进行强化处理的记录

    # 检索短时记忆
    short = memory.retrieve_short()
    print(f"\nShort-term memory:\n{short}")

    # 检索长时记忆
    long = memory.retrieve_long()
    print(f"\nLong-term memory:\n{long}")

    # 模拟做题情况进行反思纠正
    practice = {"know_name": "loop", "score": 1}
    ans = {"task2": "if statement", "task4": "no"}  # 错误识别知识 + 错误判断能否做对
    fb = memory.reflect_corrective(practice, ans)
    print(f"\nCorrective Reflection:\n{fb}")

    # 汇总反思总结（用于提示大模型）
    summary = memory.reflect_summary(fb)
    print(f"\nReflection Summary Prompt:\n{summary}")


if __name__ == "__main__":
    main()
