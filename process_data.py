import json

def process_policy_data(question: str, trajectories: list[str]) -> list[str]:
    question = question.rstrip("\n")
    all_prompts = []
    
    for trajectory in trajectories:
        steps = trajectory.strip().split('\n')
        prompts = []
        
        current_prompt = [
            {"role": "user", "content": question + "\n"}
        ]
        
        for step in steps:
            prompts.append(json.dumps({"prompt": current_prompt, "completion": [{"role": "assistant", "content": step + "\n"}]}))
            current_prompt = [{"role": "user", "content": question + "\n" + "\n".join([c["completion"][0]["content"].strip() for c in json.loads("[" + ",".join(prompts) + "]")]) + "\n"}]
        
        all_prompts.extend(prompts)
    
    return all_prompts


def process_value_data(question: str, trajectory_label_pairs: list[tuple[str, float]]) -> list[str]:
    question = question.rstrip("\n")
    all_prompts = []
    
    for trajectory, label in trajectory_label_pairs:
        steps = trajectory.strip().split('\n')
        prompts = []
        
        current_prompt = [
            {"role": "user", "content": question + "\n"}
        ]

        for step in steps:
            prompt_data = {
                "prompt": current_prompt,
                "completion": [{"role": "assistant", "content": label}]
            }
            
            current_prompt = [{"role": "user", "content": question + "\n" + "\n".join(step.strip() for step in steps[:steps.index(step)+1]) + "\n"}]
            prompts.append(json.dumps(prompt_data))
        
        all_prompts.extend(prompts)
    
    return all_prompts

# Example usage
#question = "3 7 11 12"
#trajectory1 = """3+11=14 (left: 7, 12, 14)
#7/14=0.5 (left: 12, 0.5)
#12/0.5=24.0 (left: 24.0)
#The solution is: 12/(7/(3+11)) = 24.
#"""
#trajectory2 = """3+11=14 (left: 7, 12, 14)
#7/14=0.5 (left: 12, 0.5)
#12/0.5=24.0 (left: 24.0)
#The solution is: 12/(7/(3+11)) = 24.
#"""


#output = process_policy_data(question, [trajectory1, trajectory2])


#question = "3 7 11 12"
#state1 = "3-11=-8 (left: -8, 7, 12)\n12-7=5 (left: -8, 5)\n-8*5=-40 (left: -40)\nThe solution is: (3-11)*(12-7) = -40.\n"
#state2 = "3+11=14 (left: 7, 12, 14)\n7/14=0.5 (left: 12, 0.5)\n12/0.5=24.0 (left: 24.0)\nThe solution is: 12/(7/(3+11)) = 24.\n"
#label1 = 0.0
#label2 = 1.0
#abeled_output1 = process_value_data(question, [(state1, label1), (state2, label2)])

#for i in labeled_output1:
    #print(i)

