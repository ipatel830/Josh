import torch
<<<<<<< HEAD
 
=======
from jiwer import wer, cer
 

>>>>>>> main
 
def ctc_collapse(token_ids: list[int], blank_id: int) -> list[int]:

    result = []
    prev   = None
    for tid in token_ids:
        if tid == blank_id:
            prev = None   
            continue
        if tid != prev:
            result.append(tid)
        prev = tid
    return result
 
 
def ctc_greedy_decode(logits: torch.Tensor, blank_id: int) -> list[int]:

    ids = torch.argmax(logits, dim=-1).tolist()
    return ctc_collapse(ids, blank_id)
 
 

def ids_to_text(token_ids: list[int], tokenizer) -> str:

    text = tokenizer.decode(token_ids)
    return text.replace("|", " ").strip().lower()
 
 
def label_ids_to_text(label_ids: torch.Tensor, tokenizer) -> str:

    ids  = label_ids[label_ids != 0].tolist()
    text = tokenizer.decode(ids)
    return text.replace("|", " ").strip().lower()
 
 
 
def evaluate_batch(logits: torch.Tensor, labels: torch.Tensor, tokenizer) -> tuple:
    blank_id = tokenizer.pad_token_id
    B        = logits.shape[0]
 
    predictions   = []
    ground_truths = []
 
    logits_cpu = logits.detach().cpu()
    labels_cpu = labels.detach().cpu()
 
    for i in range(B):
        # prediction: greedy CTC decode with collapsing
        collapsed = ctc_greedy_decode(logits_cpu[i], blank_id)
        pred_text = ids_to_text(collapsed, tokenizer)
        predictions.append(pred_text)
 
        # ground truth: strip padding, convert | → space
        gt_text = label_ids_to_text(labels_cpu[i], tokenizer)
        ground_truths.append(gt_text)
 
    return predictions, ground_truths
 
