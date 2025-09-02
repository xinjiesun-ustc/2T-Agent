import openai
from openai import OpenAI
from config import OPENAI_API_KEY, SIM_PARAMS
from llm_client import LLMClient
from Myprofile import build_exercise_prompt

class AgentAction:
    """
    Action Module (§4.1.4.2)
    Task1~Task2 模拟，并调用 Memory 的检索、写入与反思
    """
    def __init__(self, Userprofile,memory):
        self.Userprofile = Userprofile
        # self.Exerciseprofile = Exerciseprofile
        self.memory = memory
        self.llm = LLMClient()


    def _call_llm(self, messages):
        # print (messages)
        return self.llm.call(messages)

    def simulate_step(self, practice):  #这里的time_step 不是真实的时间戳 而是自己设定的一个答题顺序的时间戳 比如 1 2 3 4
        # 1. Memory Retrieval  对应于教师思考模式的《观察》
        short_mem = self.memory.retrieve_short()
        long_mem = self.memory.retrieve_long()

        # 2. Build prompt:  short-term + long-term memories +include profile   ###对应于教师思考模式的《解释+认知》
        prompt = self._build_prompt(practice, short_mem, long_mem)  #2个任务 包括解题思路和预测答题情况

        ##todo 这里要加一个练习题目的画像
        problem_id = practice['problem_id']
        print(problem_id)

        SysPrompt = self.Userprofile.build_User_prompt() + build_exercise_prompt(problem_id)
        messages = [
            {'role': 'system', 'content': SysPrompt}, #大模型所扮演的角色  这个里面包含了profile的相关信息
            {'role': 'user', 'content': prompt}
        ]

        # 3. LLM 生成 Task1~Task2
        resp = self._call_llm(messages)  #每调用一次LLM 就会打印一次传给LLM内容messages的打印输出

        # 4. 解析结果
        ans = {}
        for i in range(1, 3):
            tag = f'Task{i}:'
            if tag in resp:
                ans[f'task{i}'] = resp.split(tag)[1].split('\n')[0].strip()

        # 5. Memory Writing: 写入 factual  ####预测完 把当前的题目信息写入事实记忆
        record = [
            practice['content'],
            practice['concepts'],
            practice['is_correct'],
        ]
        self.memory.write_factual(record)

        # 6. Update Short-term
        self.memory.retrieve_short()

        # 7. Memory Reflection
        # 7a. Corrective Reflection
        corrective = self.memory.reflect_corrective(practice, ans) #这里只是组合成反思所需要的prompt
        # 7b. Summary Reflection
        reflect = self.memory.reflect_summary(corrective)  #这里也只是组合成总结所需要的prompt


        ## 调用 LLM 再次总结学习状态, 类似多轮对话，这里才形成了真正的总结
        messages.append({"role": "assistant",
                         "content": resp})  ####resp 上一轮的LLM返回结果
        messages.append({"role": "user",
                         "content": reflect})
        summary = self._call_llm(messages)
        
        self.memory.write_long_summary_and_kc(summary,practice['concepts'])
        # self.memory.write_know(practice)

        # 8. 返回
        return ans, resp, corrective, summary

    def _build_prompt(self, practice, short_mem, long_mem):  ###practice就是log日志文件的练习相关的记录
        """
        构建符合论文 Task1~Task2 要求的 Prompt，依次包括：
        1. Profile 提示
        2. Short-term Memory 检索结果
        3. Long-term Memory 检索结果 (强化事实、知识熟练度、学习状态)
        4. 练习内容与四项任务说明
        """
        prompt = (
            f"Assessment Exercise:\n"
            f"- Textual Content: {practice['content']}\n"  #当前题目的题干
            f"- Knowledge Concept (true): {practice['concepts']}\n"  ##当前题目的知识点
        )

        # 短期记忆展示
        if short_mem:
            prompt += "\nYour Short-term Memory (recent facts):\n"
            for idx, r in enumerate(short_mem, 1):
                prompt += (
                    f" Record {idx}: Content='{r[0]}', Concept={r[1]}, Correct={r[2]}\n"
                )
            prompt += "\n"

        # 学习状态
        if long_mem.get('learning_status'):
            prompt += (
                f"\nYour current Learning Status Summary: "
                f"{long_mem['learning_status'][-1]}\n"
            )

        # Task1: 解题思路与答案
        prompt += (
            "\nTask 1: Propose a concise problem-solving idea based on your Profile and Memories, "
            "then give a final answer.\n"
        )
        
        # Task2: 正确率预测
        prompt += (
            "\nTask 2: Predict whether you will answer correctly ('Yes' or 'No') based on the idea.\n"
        )

        # 输出格式说明
        prompt += (
            "\nOutput format exactly as:\n"
            "Task1: <your idea and final answer>\n"
            "Task2: <Yes/No>\n"
        )
        return prompt
