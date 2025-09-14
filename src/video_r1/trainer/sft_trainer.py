from trl import SFTTrainer, SFTConfig
import torch

class CustomTrainer_Distilling(SFTTrainer):
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # unpack inputs
        inputs_student = inputs["inputs_student"]
        inputs_teacher = inputs["inputs_teacher"]
        ans_start_tokens = inputs["ans_start_tokens"]
        ans_end_tokens = inputs["ans_end_tokens"]
        
        # get index of the answer
        def find_answer_indices(input_ids):
            input_ids = input_ids.cpu().tolist()[0] if isinstance(input_ids, torch.Tensor) else input_ids
            answer_indices = {
                "start": [],
                "end": [],
            }

            for i in range(len(input_ids)):
                for j in range(len(ans_start_tokens)):
                    if i + len(ans_start_tokens[j]) > len(input_ids):
                        continue
                    if all([input_ids[i+k] == ans_start_tokens[j][k] for k in range(len(ans_start_tokens[j]))]):
                        answer_indices["start"] = list(range(i, i+len(ans_start_tokens[j])))
                        
                
                for j in range(len(ans_end_tokens)):
                    if i + len(ans_end_tokens[j]) > len(input_ids):
                        continue
                    if all([input_ids[i+k] == ans_end_tokens[j][k] for k in range(len(ans_end_tokens[j]))]):
                        answer_indices["end"] = list(range(i, i+len(ans_end_tokens[j])))

            assert len(answer_indices["start"]) > 0 and len(answer_indices["end"]) > 0 and answer_indices["start"][-1] < answer_indices["end"][0], f"ERROR: answer indices are not correct: {answer_indices}, sequence length is {len(input_ids)}"
            
            answer_indices["indices_to_match"] = list(range(answer_indices["start"][-1], answer_indices["end"][0] - 1))
            return answer_indices

        student_answer_indices = find_answer_indices(inputs_student["input_ids"])
        teacher_answer_indices = find_answer_indices(inputs_teacher["input_ids"])

        # llm forward
        inputs_student["output_hidden_states"] = True
        inputs_teacher["output_hidden_states"] = True

        (ce_loss_student, outputs_student) = super().compute_loss(
            model, inputs_student, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        (ce_loss_teacher, outputs_teacher) = super().compute_loss(
            model, inputs_teacher, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        
        # calculate loss
        student_hidden_states = outputs_student["hidden_states"][1:]
        with torch.no_grad():
            teacher_hidden_states = outputs_teacher["hidden_states"][1:]
            teacher_hidden_states = [state.detach() for state in teacher_hidden_states]

        print(f"\n\nStudent: hidden states num - {len(student_hidden_states)}, hidden states shapes - {student_hidden_states[0].shape}, answer indices - {student_answer_indices}, seq length - {len(inputs_student['input_ids'][0])}\nTeacher: hidden states num - {len(teacher_hidden_states)}, hidden states shapes - {teacher_hidden_states[0].shape}, answer indices - {teacher_answer_indices}, seq length - {len(inputs_teacher['input_ids'][0])}\n\n")
        
        kd_loss = []
        for layer in range(len(student_hidden_states)):
            for token_s, token_t in zip(student_answer_indices["indices_to_match"], teacher_answer_indices["indices_to_match"]):
                student_hidden_state = student_hidden_states[layer][0, token_s, :].squeeze()
                teacher_hidden_state = teacher_hidden_states[layer][0, token_t, :].squeeze()

                knowledge_distillation_loss = torch.nn.functional.smooth_l1_loss(student_hidden_state, teacher_hidden_state)
                knowledge_distillation_loss_scaled = knowledge_distillation_loss / (teacher_hidden_state.std() + 1e-6)

                # print(f"Layer: {layer}, token_s: {token_s}, token_t: {token_t}, kd loss: {knowledge_distillation_loss}, scale: {teacher_hidden_state.std()}, kd loss scaled: {knowledge_distillation_loss_scaled}")
                kd_loss.append(knowledge_distillation_loss_scaled)
        
        kd_loss_avg = sum(kd_loss) / len(kd_loss)
        total_loss = ce_loss_student + ce_loss_teacher + 10 * kd_loss_avg
        
        # Log losses to tensorboard
        if hasattr(self, "state"):
            step = self.state.global_step
            self.log({
                "train/ce_loss_student": ce_loss_student.item(),
                "train/ce_loss_teacher": ce_loss_teacher.item(),
                "train/kd_loss": kd_loss_avg.item() * 10,  # Multiply by 10 since we use this scaling in total_loss
                "train/total_loss": total_loss.item()
            })

        print(f"ce for student: {ce_loss_student}, ce for teacher: {ce_loss_teacher}, kd length: {len(kd_loss)}, kd sum: {sum(kd_loss)}, kd: {kd_loss_avg}, total loss: {total_loss}")

        return total_loss



class CustomTrainer_Seperate_Student(SFTTrainer):
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # unpack inputs
        inputs_student_cot = inputs["inputs_student_cot"]
        inputs_student_ans = inputs["inputs_student_ans"]
        
        (ce_loss_student_cot, outputs_student_cot) = super().compute_loss(
            model, inputs_student_cot, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        (ce_loss_student_ans, outputs_student_ans) = super().compute_loss(
            model, inputs_student_ans, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        
        total_loss = (ce_loss_student_cot + ce_loss_student_ans) / 2
        
        # Log losses to tensorboard
        if hasattr(self, "state"):
            step = self.state.global_step
            self.log({
                "train/ce_loss_student_cot": ce_loss_student_cot.item(),
                "train/ce_loss_student_ans": ce_loss_student_ans.item(),
                "train/total_loss": total_loss.item()
            })

        print(f"ce for student_cot: {ce_loss_student_cot}, ce for student_ans: {ce_loss_student_ans}, total loss: {total_loss}")

        return total_loss
        
        
        
class CustomTrainer_Seperate_Distilling(SFTTrainer):
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # unpack inputs
        inputs_student_cot = inputs["inputs_student_cot"]
        inputs_student_ans = inputs["inputs_student_ans"]
        inputs_teacher = inputs["inputs_teacher"]
        ans_start_tokens = inputs["ans_start_tokens"]
        ans_end_tokens = inputs["ans_end_tokens"]
        
        # get index of the answer
        def find_answer_indices(input_ids):
            input_ids = input_ids.cpu().tolist()[0] if isinstance(input_ids, torch.Tensor) else input_ids
            answer_indices = {
                "start": [],
                "end": [],
            }

            for i in range(len(input_ids)):
                for j in range(len(ans_start_tokens)):
                    if i + len(ans_start_tokens[j]) > len(input_ids):
                        continue
                    if all([input_ids[i+k] == ans_start_tokens[j][k] for k in range(len(ans_start_tokens[j]))]):
                        answer_indices["start"] = list(range(i, i+len(ans_start_tokens[j])))
                        
                
                for j in range(len(ans_end_tokens)):
                    if i + len(ans_end_tokens[j]) > len(input_ids):
                        continue
                    if all([input_ids[i+k] == ans_end_tokens[j][k] for k in range(len(ans_end_tokens[j]))]):
                        answer_indices["end"] = list(range(i, i+len(ans_end_tokens[j])))

            assert len(answer_indices["start"]) > 0 and len(answer_indices["end"]) > 0 and answer_indices["start"][-1] < answer_indices["end"][0], f"ERROR: answer indices are not correct: {answer_indices}, sequence length is {len(input_ids)}"
            
            answer_indices["indices_to_match"] = list(range(answer_indices["start"][-1], answer_indices["end"][0] - 1))
            return answer_indices

        student_answer_indices = find_answer_indices(inputs_student_ans["input_ids"])
        teacher_answer_indices = find_answer_indices(inputs_teacher["input_ids"])

        # llm forward
        inputs_student_ans["output_hidden_states"] = True
        inputs_teacher["output_hidden_states"] = True

        (ce_loss_student_cot, outputs_student_cot) = super().compute_loss(
            model, inputs_student_cot, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        (ce_loss_student_ans, outputs_student_ans) = super().compute_loss(
            model, inputs_student_ans, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        (ce_loss_teacher, outputs_teacher) = super().compute_loss(
            model, inputs_teacher, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        
        # calculate loss
        student_hidden_states = outputs_student_ans["hidden_states"][1:]
        with torch.no_grad():
            teacher_hidden_states = outputs_teacher["hidden_states"][1:]
            teacher_hidden_states = [state.detach() for state in teacher_hidden_states]

        print(f"\n\nStudent: hidden states num - {len(student_hidden_states)}, hidden states shapes - {student_hidden_states[0].shape}, answer indices - {student_answer_indices}, seq length - {len(inputs_student_ans['input_ids'][0])}\nTeacher: hidden states num - {len(teacher_hidden_states)}, hidden states shapes - {teacher_hidden_states[0].shape}, answer indices - {teacher_answer_indices}, seq length - {len(inputs_teacher['input_ids'][0])}\n\n")
        
        kd_loss = []
        for layer in range(len(student_hidden_states)):
            for token_s, token_t in zip(student_answer_indices["indices_to_match"], teacher_answer_indices["indices_to_match"]):
                student_hidden_state = student_hidden_states[layer][0, token_s, :].squeeze()
                teacher_hidden_state = teacher_hidden_states[layer][0, token_t, :].squeeze()

                knowledge_distillation_loss = torch.nn.functional.smooth_l1_loss(student_hidden_state, teacher_hidden_state)
                knowledge_distillation_loss_scaled = knowledge_distillation_loss / (teacher_hidden_state.std() + 1e-6)

                # print(f"Layer: {layer}, token_s: {token_s}, token_t: {token_t}, kd loss: {knowledge_distillation_loss}, scale: {teacher_hidden_state.std()}, kd loss scaled: {knowledge_distillation_loss_scaled}")
                kd_loss.append(knowledge_distillation_loss_scaled)
        
        kd_loss_avg = sum(kd_loss) / len(kd_loss)
        total_loss = (ce_loss_student_cot + ce_loss_student_ans) / 2 + ce_loss_teacher + 10 * kd_loss_avg
        
        # Log losses to tensorboard
        if hasattr(self, "state"):
            step = self.state.global_step
            self.log({
                "train/ce_loss_student_cot": ce_loss_student_cot.item(),
                "train/ce_loss_student_ans": ce_loss_student_ans.item(),
                "train/ce_loss_teacher": ce_loss_teacher.item(),
                "train/kd_loss": kd_loss_avg.item() * 10,  # Multiply by 10 since we use this scaling in total_loss
                "train/total_loss": total_loss.item()
            })

        print(f"ce for student_cot: {ce_loss_student_cot}, ce for student_ans: {ce_loss_student_ans}, ce for teacher: {ce_loss_teacher}, kd length: {len(kd_loss)}, kd sum: {sum(kd_loss)}, kd: {kd_loss_avg}, total loss: {total_loss}")

        return total_loss
        

        
        



