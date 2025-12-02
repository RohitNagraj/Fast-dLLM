import torch
import numpy as np
import gradio as gr
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM
import time
import re


def printdebug(*args, **kwargs):
    print(*args, **kwargs)
    import sys
    sys.stdout.flush()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, 
                                  torch_dtype=torch.bfloat16).to(device)

# Constants
MASK_TOKEN = "[MASK]"
MASK_ID = 126336  # The token ID of [MASK] in LLaDA
question_gsm8k = '''Question: Jen and Tyler are gymnasts practicing flips. Jen is practicing the triple-flip while Tyler is practicing the double-flip. Jen did sixteen triple-flips during practice. Tyler flipped in the air half the number of times Jen did. How many double-flips did Tyler do?
Answer: Jen did 16 triple-flips, so she did 16 * 3 = <<16*3=48>>48 flips.
Tyler did half the number of flips, so he did 48 / 2 = <<48/2=24>>24 flips.
A double flip has two flips, so Tyler did 24 / 2 = <<24/2=12>>12 double-flips.
#### 12

Question: Four people in a law firm are planning a party. Mary will buy a platter of pasta for $20 and a loaf of bread for $2. Elle and Andrea will split the cost for buying 4 cans of soda which cost $1.50 each, and chicken wings for $10. Joe will buy a cake that costs $5. How much more will Mary spend than the rest of the firm put together?
Answer: Mary will spend $20 + $2 = $<<20+2=22>>22.
Elle and Andrea will spend $1.5 x 4 = $<<1.5*4=6>>6 for the soda.
Elle and Andrea will spend $6 + $10 = $<<6+10=16>>16 for the soda and chicken wings.
Elle, Andrea, and Joe together will spend $16 + $5 = $<<16+5=21>>21.
So, Mary will spend $22 - $21 = $<<22-21=1>>1 more than all of them combined.
#### 1

Question: A charcoal grill burns fifteen coals to ash every twenty minutes of grilling. The grill ran for long enough to burn three bags of coals. Each bag of coal contains 60 coals. How long did the grill run?
Answer: The grill burned 3 * 60 = <<3*60=180>>180 coals.
It takes 20 minutes to burn 15 coals, so the grill ran for 180 / 15 * 20 = <<180/15*20=240>>240 minutes.
#### 240

Question: A bear is preparing to hibernate for the winter and needs to gain 1000 pounds. At the end of summer, the bear feasts on berries and small woodland animals. During autumn, it devours acorns and salmon. It gained a fifth of the weight it needed from berries during summer, and during autumn, it gained twice that amount from acorns. Salmon made up half of the remaining weight it had needed to gain. How many pounds did it gain eating small animals?
Answer: The bear gained 1 / 5 * 1000 = <<1/5*1000=200>>200 pounds from berries.
It gained 2 * 200 = <<2*200=400>>400 pounds from acorns.
It still needed 1000 - 200 - 400 = <<1000-200-400=400>>400 pounds.
Thus, it gained 400 / 2 = <<400/2=200>>200 pounds from salmon.
Therefore, the bear gained 400 - 200 = <<400-200=200>>200 pounds from small animals.
#### 200

Question: Brendan can cut 8 yards of grass per day, he bought a lawnmower and it helped him to cut more yards by Fifty percent per day. How many yards will Brendan be able to cut after a week?
Answer: The additional yard Brendan can cut after buying the lawnmower is 8 x 0.50 = <<8*0.50=4>>4 yards.
So, the total yards he can cut with the lawnmower is 8 + 4 = <<8+4=12>>12.
Therefore, the total number of yards he can cut in a week is 12 x 7 = <<12*7=84>>84 yards.
#### 84

Question: Skyler has 100 hats on his hand with the colors red, blue, and white. Half of the hats are red, 3/5 of the remaining hats are blue, and the rest are white. How many white hats does Skyler have?'''

def parse_constraints(constraints_text):
    """Parse constraints in format: 'position:word, position:word, ...'"""
    constraints = {}
    if not constraints_text:
        return constraints
        
    parts = constraints_text.split(',')
    for part in parts:
        if ':' not in part:
            continue
        pos_str, word = part.split(':', 1)
        try:
            pos = int(pos_str.strip())
            word = word.strip()
            if word and pos >= 0:
                constraints[pos] = word
        except ValueError:
            continue
    
    return constraints

def format_chat_history(history):
    """
    Format chat history for the LLaDA model
    
    Args:
        history: List of [user_message, assistant_message] pairs
        
    Returns:
        Formatted conversation for the model
    """
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:  # Skip if None (for the latest user message)
            messages.append({"role": "assistant", "content": assistant_msg})
    
    return messages

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature <= 0:
        return logits
        
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def generate_response_with_visualization_cache_and_parallel(model, tokenizer, device, messages, gen_length=64, steps=32, 
                                         constraints=None, temperature=0.0, block_length=32,
                                         remasking='low_confidence', threshold=0.9):
    """
    Generate text with LLaDA model with visualization using the same sampling as in generate.py
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        gen_length: Length of text to generate
        steps: Number of denoising steps
        constraints: Dictionary mapping positions to words
        temperature: Sampling temperature
        block_length: Block length for semi-autoregressive generation
        remasking: Remasking strategy ('low_confidence' or 'random')
        
    Returns:
        List of visualization states showing the progression and final text
    """
    START = torch.cuda.Event(enable_timing=True)
    END = torch.cuda.Event(enable_timing=True)
    MODEL_START = torch.cuda.Event(enable_timing=True)
    MODEL_END = torch.cuda.Event(enable_timing=True)
    MODEL_TIME = []
    START.record()

    profile = True
    profile = False
    if profile:
        _start = torch.cuda.Event(enable_timing=True)
        _end = torch.cuda.Event(enable_timing=True)

    # Process constraints
    if constraints is None:
        constraints = {}
        
    # Convert any string constraints to token IDs
    processed_constraints = {}
    for pos, word in constraints.items():
        tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        for i, token_id in enumerate(tokens):
            processed_constraints[pos + i] = token_id
    
    # Prepare the prompt using chat template
    chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(chat_input)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    # For generation
    prompt_length = input_ids.shape[1]
    
    # Initialize the sequence with masks for the response part
    x = torch.full((1, prompt_length + gen_length), MASK_ID, dtype=torch.long).to(device)
    x[:, :prompt_length] = input_ids.clone()
    
    # Initialize visualization states for the response part
    visualization_states = []
    
    # Add initial state (all masked)
    initial_state = [(MASK_TOKEN, "#444444") for _ in range(gen_length)]
    visualization_states.append(initial_state)
    
    # Apply constraints to the initial state
    for pos, token_id in processed_constraints.items():
        absolute_pos = prompt_length + pos
        if absolute_pos < x.shape[1]:
            x[:, absolute_pos] = token_id
    
    # Ensure block_length is valid
    if block_length > gen_length:
        block_length = gen_length
    
    # Calculate number of blocks
    num_blocks = gen_length // block_length
    if gen_length % block_length != 0:
        num_blocks += 1
    
    # Adjust steps per block
    steps_per_block = steps // num_blocks
    if steps_per_block < 1:
        steps_per_block = 1
    
    # Process each block
    for num_block in range(num_blocks):
        current_block_start = prompt_length + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == MASK_ID)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        if profile:
            _start.record()

        MODEL_START.record()
        output = model(x, use_cache=True)
        MODEL_END.record()
        torch.cuda.synchronize()
        MODEL_TIME.append(MODEL_START.elapsed_time(MODEL_END))

        if profile:
            _end.record()
            torch.cuda.synchronize()
            print(f"### Model inference [initial] time for block {num_block}: {_start.elapsed_time(_end)} ms")
        
        past_key_values = output.past_key_values

        mask_index = (x == MASK_ID)
        mask_index[:, current_block_end:] = 0
        x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        # Create visualization state only for the response part
        current_state = []
        for i in range(gen_length):
            pos = prompt_length + i  # Absolute position in the sequence
            
            if x[0, pos] == MASK_ID:
                # Still masked
                current_state.append((MASK_TOKEN, "#444444"))  # Dark gray for masks
            else:
                # Previously revealed
                token = tokenizer.decode([x[0, pos].item()], skip_special_tokens=True)
                current_state.append((token, "#6699CC"))  # Light blue
        
        visualization_states.append(current_state)
        i = 1
        while True:
            # print("###", x.shape, current_block_start, current_block_end)
            mask_index = (x[:, current_block_start:] == MASK_ID)
            mask_index[:, block_length:] = 0

            if profile:
                _start.record()

            MODEL_START.record()
            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits
            MODEL_END.record()
            torch.cuda.synchronize()
            MODEL_TIME.append(MODEL_START.elapsed_time(MODEL_END))

            if profile:
                _end.record()
                torch.cuda.synchronize()
                print(f" ### Model inference [logits] time for block {num_block}, step {i}: {_start.elapsed_time(_end)} ms")

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                            x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            # Create visualization state only for the response part
            current_state = []
            for i in range(gen_length):
                pos = prompt_length + i  # Absolute position in the sequence
                
                if x[0, pos] == MASK_ID:
                    # Still masked
                    current_state.append((MASK_TOKEN, "#444444"))  # Dark gray for masks
                else:
                    # Previously revealed
                    token = tokenizer.decode([x[0, pos].item()], skip_special_tokens=True)
                    current_state.append((token, "#6699CC"))  # Light blue
            
            visualization_states.append(current_state)
            if (x[:, current_block_start:current_block_end] == MASK_ID).sum() == 0:
                break
            i += 1

    END.record()
    torch.cuda.synchronize()
    printdebug(f"[fast] Total generation time: {START.elapsed_time(END)} ms")
    printdebug(f"[fast] \t\tModel execution time: {sum(MODEL_TIME)} ms over {len(MODEL_TIME)} calls")
    printdebug(f"[fast] \t\tMean model time per call: {sum(MODEL_TIME)/len(MODEL_TIME)} ms")
    printdebug(f"[fast] \t\tOther time: {START.elapsed_time(END) - sum(MODEL_TIME)} ms")

    # Extract final text (just the assistant's response)
    response_tokens = x[0, prompt_length:]
    final_text = tokenizer.decode(response_tokens, 
                               skip_special_tokens=True,
                               clean_up_tokenization_spaces=True)
    
    return visualization_states, final_text


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def generate_response_with_visualization(model, tokenizer, device, messages, gen_length=64, steps=32, 
                                         constraints=None, temperature=0.0, block_length=32,
                                         remasking='low_confidence'):
    """
    Generate text with LLaDA model with visualization using the same sampling as in generate.py
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        gen_length: Length of text to generate
        steps: Number of denoising steps
        constraints: Dictionary mapping positions to words
        temperature: Sampling temperature
        block_length: Block length for semi-autoregressive generation
        remasking: Remasking strategy ('low_confidence' or 'random')
        
    Returns:
        List of visualization states showing the progression and final text
    """
    START = torch.cuda.Event(enable_timing=True)
    END = torch.cuda.Event(enable_timing=True)
    MODEL_START = torch.cuda.Event(enable_timing=True)
    MODEL_END = torch.cuda.Event(enable_timing=True)
    MODEL_TIME = []
    START.record()

    profile = True
    profile = False
    if profile:
        _start = torch.cuda.Event(enable_timing=True)
        _end = torch.cuda.Event(enable_timing=True)
        _block_start = torch.cuda.Event(enable_timing=True)
        _block_end = torch.cuda.Event(enable_timing=True)
        _tmp_start = torch.cuda.Event(enable_timing=True)
        _tmp_end = torch.cuda.Event(enable_timing=True)
        _start.record()

    # Process constraints
    if constraints is None:
        constraints = {}
        
    # Convert any string constraints to token IDs
    processed_constraints = {}
    for pos, word in constraints.items():
        tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        for i, token_id in enumerate(tokens):
            processed_constraints[pos + i] = token_id
    
    # Prepare the prompt using chat template
    # printdebug(f"messages_input: {messages}")

    chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # printdebug(f"chat_input: {chat_input}")

    input_ids = tokenizer(chat_input)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    # For generation
    prompt_length = input_ids.shape[1]
    
    # printdebug(f"mask_id: {MASK_ID} =? 126336, prompt_length: {prompt_length}, gen_length: {gen_length}")

    # Initialize the sequence with masks for the response part
    x = torch.full((1, prompt_length + gen_length), MASK_ID, dtype=torch.long).to(device)
    x[:, :prompt_length] = input_ids.clone()
    
    # Initialize visualization states for the response part
    visualization_states = []
    
    # Add initial state (all masked)
    initial_state = [(MASK_TOKEN, "#444444") for _ in range(gen_length)]
    visualization_states.append(initial_state)

    # printdebug(f"no constraints? {len(processed_constraints)}")
    
    # Apply constraints to the initial state
    for pos, token_id in processed_constraints.items():
        absolute_pos = prompt_length + pos
        if absolute_pos < x.shape[1]:
            x[:, absolute_pos] = token_id
    
    # Mark prompt positions to exclude them from masking during classifier-free guidance
    prompt_index = (x != MASK_ID)

    # printdebug(f"prompt_index: {prompt_index}")
    
    # Ensure block_length is valid
    if block_length > gen_length:
        block_length = gen_length
    
    # Calculate number of blocks
    num_blocks = gen_length // block_length
    if gen_length % block_length != 0:
        num_blocks += 1
    
    # Adjust steps per block
    steps_per_block = steps // num_blocks
    if steps_per_block < 1:
        steps_per_block = 1

    # printdebug(f"block_length: {block_length}")
    # printdebug(f"steps_per_block: {steps_per_block}, steps: {steps}, num_blocks: {num_blocks}")
    
    if profile:
        _end.record()
        torch.cuda.synchronize()
        printdebug(f"Preparation time: {_start.elapsed_time(_end)} ms")

    # Process each block
    for num_block in range(num_blocks):

        if profile:
            _block_start.record()

        # Calculate the start and end indices for the current block
        block_start = prompt_length + num_block * block_length
        block_end = min(prompt_length + (num_block + 1) * block_length, x.shape[1])

        # printdebug(f"\tnum_block: {num_block}, block_start: {block_start}, block_end: {block_end}")
        
        # Get mask indices for the current block
        block_mask_index = (x[:, block_start:block_end] == MASK_ID)

        # printdebug(f"\tblock_mask_index: {block_mask_index}")
        
        # Skip if no masks in this block
        if not block_mask_index.any():
            continue
        
        # Calculate number of tokens to unmask at each step
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # printdebug(f"doing transfer tokens: In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.")
        # printdebug(f"\tnum_transfer_tokens: {num_transfer_tokens}")

        # Process each step
        for i in range(steps_per_block):


            if profile:
                _start.record()

            # printdebug(f"\t\tstep: {i}/{steps_per_block}")

            # Get all mask positions in the current sequence
            mask_index = (x == MASK_ID)

            # printdebug(f"\t\tmask_index: {mask_index}")
            
            # Skip if no masks
            if not mask_index.any():
                break

            if profile:
                _end.record()
                torch.cuda.synchronize()
                printdebug(f"\t\tStep preparation time: {_start.elapsed_time(_end)} ms")
                _start.record()
            
            # Get logits from model
            MODEL_START.record()
            logits = model(x).logits
            MODEL_END.record()
            torch.cuda.synchronize()
            MODEL_TIME.append(MODEL_START.elapsed_time(MODEL_END))

            if profile:
                _end.record()
                torch.cuda.synchronize()
                printdebug(f"\t\tModel inference time: {_start.elapsed_time(_end)} ms")
                _start.record()

            # torch.set_printoptions(threshold=float('inf'))
            # printdebug(model(x))
            # printdebug(f"logits: {logits}")
            # printdebug()
            
            # printdebug("\t\tApplying Gumbel noise and sampling... The Gumbel max is a method for sampling categorical distributions. ... but reduces generation quality. Thus, we use float64.")

            # Apply Gumbel noise for sampling
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if profile:
                _end.record()
                torch.cuda.synchronize()
                printdebug(f"\t\tGumbel noise and sampling time: {_start.elapsed_time(_end)} ms")
                _start.record()

            # printdebug(f"\t\tlogits_with_noise: {logits_with_noise}")
            # printdebug(f"\t\tx0 (sampled tokens): {x0}")

            # printdebug(f"\t\tremasking strategy: {remasking}")
            
            # Calculate confidence scores for remasking
            if remasking == 'low_confidence':
                if profile:
                    printdebug(f"\t\t\t=> type of logits: {logits.dtype}")
                    _tmp_start.record()

                # p = F.softmax(logits, dim=-1)
                p = F.softmax(logits.to(torch.float64), dim=-1)

                if profile:
                    _tmp_end.record()
                    torch.cuda.synchronize()
                    printdebug(f"\t\t\tSoftmax calculation time: {_tmp_start.elapsed_time(_tmp_end)} ms")
                    _tmp_start.record()

                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l

                if profile:
                    _tmp_end.record()
                    torch.cuda.synchronize()
                    printdebug(f"\t\t\tConfidence score gathering time: {_tmp_start.elapsed_time(_tmp_end)} ms")
                    _tmp_start.record()

            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")
            

            if profile:
                _end.record()
                torch.cuda.synchronize()
                printdebug(f"\t\tConfidence score calculation time: {_start.elapsed_time(_end)} ms")
                _start.record()

            # printdebug(f"\t\tx0_p (confidence scores): {x0_p}")

            # Don't consider positions beyond the current block
            x0_p[:, block_end:] = -float('inf')

            # printdebug(f"\t\tx0_p after masking beyond block_end: {x0_p}")
            
            # Apply predictions where we have masks
            old_x = x.clone()
            x0 = torch.where(mask_index, x0, x)

            if profile:
                _end.record()
                torch.cuda.synchronize()
                printdebug(f"\t\tApply predictions where we have masks time: {_start.elapsed_time(_end)} ms")
                _start.record()

            # printdebug(f"\t\tx0 after applying mask_index: {x0}")

            confidence = torch.where(mask_index, x0_p, -float('inf'))

            # printdebug(f"\t\tconfidence after applying mask_index: {confidence}")
            # printdebug(f"\t\tnow select transfer index...")

            # Select tokens to unmask based on confidence
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

            # printdebug(f"\t\ttransfer_index initialized: {transfer_index}")

            if profile:
                _end.record()
                torch.cuda.synchronize()
                printdebug(f"\t\tFind transfer index time: {_start.elapsed_time(_end)} ms")
                _start.record()

            if profile:
                _tmp_start.record()

            for j in range(confidence.shape[0]):

                # printdebug(f"\t\t\tprocessing batch (?) item: {j}, in step {i}")

                # Only consider positions within the current block for unmasking
                block_confidence = confidence[j, block_start:block_end]
                if i < steps_per_block - 1:  # Not the last step
                    # Take top-k confidences

                    # printdebug(f"block_confidence: {block_confidence}")
                    # printdebug(f"num_transfer_tokens: {num_transfer_tokens}")
                    # printdebug(f"num_transfer_tokens[{j}, {i}]: {num_transfer_tokens[j, i].item()}, block_confidence.numel(): {block_confidence.numel()}")
                    # printdebug()

                    _, select_indices = torch.topk(block_confidence, 
                                                  k=min(num_transfer_tokens[j, i].item(), 
                                                       block_confidence.numel()))
                    # Adjust indices to global positions
                    select_indices = select_indices + block_start
                    transfer_index[j, select_indices] = True
                else:  # Last step - unmask everything remaining
                    transfer_index[j, block_start:block_end] = mask_index[j, block_start:block_end]

                # printdebug(f"\t\t\tselected indices for transfer: {select_indices if i < steps_per_block - 1 else 'all remaining masks'}")
                # printdebug(f"\t\t\ttransfer_index after selection: {transfer_index}")
            
            if profile:
                _tmp_end.record()
                torch.cuda.synchronize()
                printdebug(f"\t\t\tSelecting transfer indices time (topk): {_tmp_start.elapsed_time(_tmp_end)} ms")
                _tmp_start.record()

            # Apply the selected tokens
            x = torch.where(transfer_index, x0, x)

            if profile:
                _tmp_end.record()
                torch.cuda.synchronize()
                printdebug(f"\t\t\tApplying transfer index time (where transfer_index): {_tmp_start.elapsed_time(_tmp_end)} ms")

            # printdebug(f"\t\tx after applying transfer_index: {x}")
            
            # printdebug(f"\t\tApplying constraints to ensure they are maintained...")
            # printdebug(f"\t\tprocessed_constraints: {processed_constraints} (== ?)")
            # Ensure constraints are maintained
            for pos, token_id in processed_constraints.items():
                absolute_pos = prompt_length + pos
                if absolute_pos < x.shape[1]:
                    x[:, absolute_pos] = token_id

            if profile:
                _end.record()
                torch.cuda.synchronize()
                printdebug(f"\t\tupdate token time: {_start.elapsed_time(_end)} ms")
            
            # Create visualization state only for the response part
            current_state = []
            for i in range(gen_length):
                pos = prompt_length + i  # Absolute position in the sequence
                
                if x[0, pos] == MASK_ID:
                    # Still masked
                    current_state.append((MASK_TOKEN, "#444444"))  # Dark gray for masks
                else:
                    # Previously revealed
                    token = tokenizer.decode([x[0, pos].item()], skip_special_tokens=True)
                    current_state.append((token, "#6699CC"))  # Light blue
            
            visualization_states.append(current_state)

            # printdebug(f"\t\tcurrent_state: {current_state}")
            # printdebug(f"length of visualization_states: {len(visualization_states)}")
        
        if profile:
            _block_end.record()
            torch.cuda.synchronize()
            printdebug(f"Block {num_block} time: {_block_start.elapsed_time(_block_end)} ms")
    
    END.record()
    torch.cuda.synchronize()
    printdebug(f"[ori]  Total generation time: {START.elapsed_time(END)} ms")
    printdebug(f"[ori]  \t\tModel execution time: {sum(MODEL_TIME)} ms over {len(MODEL_TIME)} calls")
    printdebug(f"[ori]  \t\tMean model time per call: {sum(MODEL_TIME)/len(MODEL_TIME)} ms")
    printdebug(f"[ori]  \t\tOther time: {START.elapsed_time(END) - sum(MODEL_TIME)} ms")

    # Extract final text (just the assistant's response)
    response_tokens = x[0, prompt_length:]
    final_text = tokenizer.decode(response_tokens, 
                               skip_special_tokens=True,
                               clean_up_tokenization_spaces=True)

    # printdebug(f"response_tokens: {response_tokens}")
    # printdebug(f"final_text: {final_text}")
    
    return visualization_states, final_text

css = '''
.category-legend{display:none}
.message, .bubble, .chatbot .message, .chatbot .bubble {
    max-width: 80% !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
    box-sizing: border-box !important;
}
'''
def create_chatbot_demo():
    with gr.Blocks(css=css) as demo:
        gr.Markdown("# Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding")
        gr.Markdown("[code](https://github.com/NVlabs/Fast-dLLM), [project page](https://nvlabs.github.io/Fast-dLLM/)")
        
        # STATE MANAGEMENT
        chat_history_baseline = gr.State([])
        chat_history_cache = gr.State([])
        
        # UI COMPONENTS
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_ui = gr.Chatbot(label="Conversation", height=500)
            with gr.Column(scale=2):
                output_vis = gr.HighlightedText(
                    label="Denoising Process Visualization",
                    combine_adjacent=False,
                    show_legend=True,
                )
                generation_time = gr.Textbox(
                    label="Generation Time",
                    value="0.00s",
                    interactive=False
                )
                throughput = gr.Textbox(
                    label="Generation Speed",
                    value="0.00 tokens/s",
                    interactive=False
                )
        
        # Add separator line
        gr.Markdown("---")
        
        # Duplicate conversation interface
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_ui_copy = gr.Chatbot(label="Conversation (Accelerated)", height=500)
            with gr.Column(scale=2):
                output_vis_copy = gr.HighlightedText(
                    label="Denoising Process Visualization",
                    combine_adjacent=False,
                    show_legend=True,
                )
                generation_time_copy = gr.Textbox(
                    label="Generation Time",
                    value="0.00s",
                    interactive=False
                )
                throughput_copy = gr.Textbox(
                    label="Generation Speed",
                    value="0.00 tokens/s",
                    interactive=False
                )
        # Move input area below the duplicate conversation interface
        with gr.Group():
            user_input = gr.Textbox(
                label="Your Message", 
                placeholder="Type your message here...",
                show_label=False
            )
            send_btn = gr.Button("Send")
            constraints_input = gr.Textbox(
                label="Word Constraints", 
                info="This model allows for placing specific words at specific positions using 'position:word' format. Example: 1st word once, 6th word 'upon' and 11th word 'time', would be: '0:Once, 5:upon, 10:time",
                placeholder="0:Once, 5:upon, 10:time",
                value=""
            )
            gr.Examples(
                examples=[
                    [question_gsm8k]
                ],
                inputs=user_input,
                label="Example Inputs"
            )
        
        # Advanced generation settings
        with gr.Accordion("Generation Settings", open=False):
            with gr.Row():
                gen_length = gr.Slider(
                    minimum=64, maximum=1024, value=256, step=64,
                    label="Generation Length"
                )
                steps = gr.Slider(
                    minimum=8, maximum=1024, value=256, step=4,
                    label="Denoising Steps"
                )
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0, step=0.1,
                    label="Temperature"
                )
                threshold = gr.Slider(
                    minimum=0.5, maximum=1.0, value=0.9, step=0.1,
                    label="Threshold"
                )
            with gr.Row():
                block_length = gr.Slider(
                    minimum=8, maximum=128, value=32, step=8,
                    label="Block Length"
                )
                remasking_strategy = gr.Radio(
                    choices=["low_confidence", "random"],
                    value="low_confidence",
                    label="Remasking Strategy"
                )
            with gr.Row():
                visualization_delay = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.1, step=0.1,
                    label="Visualization Delay (seconds)"
                )
        
        # Current response text box (hidden)
        current_response = gr.Textbox(
            label="Current Response",
            placeholder="The assistant's response will appear here...",
            lines=3,
            visible=False
        )
        
        # Clear button
        clear_btn = gr.Button("Clear Conversation")
        
        # HELPER FUNCTIONS
        def add_message(history, message, response):
            """Add a message pair to the history and return the updated history"""
            history = history.copy()
            history.append([message, response])
            return history
            
        def user_message_submitted(message, history_baseline, history_cache, gen_length, steps, constraints, delay):
            """Process a submitted user message"""
            # Skip empty messages
            if not message.strip():
                # Return current state unchanged
                history_baseline_for_display = history_baseline.copy()
                history_cache_for_display = history_cache.copy()
                return history_baseline, history_cache, history_baseline_for_display, history_cache_for_display, "", [], [], "", "0.00s", "0.00 tokens/s", "0.00s", "0.00 tokens/s"
                
            # Add user message to both histories
            history_baseline = add_message(history_baseline, message, None)
            history_cache = add_message(history_cache, message, None)
            
            # Format for display - temporarily show user message with empty response
            history_baseline_for_display = history_baseline.copy()
            history_cache_for_display = history_cache.copy()
            
            # Clear the input
            message_out = ""
            
            # Return immediately to update UI with user message
            return history_baseline, history_cache, history_baseline_for_display, history_cache_for_display, message_out, [], [], "", "0.00s", "0.00 tokens/s", "0.00s", "0.00 tokens/s"
            
        def bot_response(history_baseline, history_cache, gen_length, steps, constraints, delay, temperature, block_length, remasking, threshold):
            """Generate bot response for the latest message"""
            if not history_baseline or not history_cache:
                return history_baseline, history_cache, [], [], "", "0.00s", "0.00 tokens/s", "0.00s", "0.00 tokens/s"
                
            # Get the last user message
            last_user_message = history_baseline[-1][0]
            
            try:
                # Format all messages except the last one (which has no response yet)
                messages = format_chat_history(history_baseline[:-1])
                
                # Add the last user message
                messages.append({"role": "user", "content": last_user_message})
                
                # Parse constraints
                parsed_constraints = parse_constraints(constraints)
                
                # Start timing for baseline
                start_time = time.time()
                
                # Generate response with visualization for baseline
                vis_states, response_text = generate_response_with_visualization(
                    model, tokenizer, device, 
                    messages, 
                    gen_length=gen_length, 
                    steps=steps,
                    constraints=parsed_constraints,
                    temperature=temperature,
                    block_length=block_length,
                    remasking=remasking,
                )
                
                # Calculate generation time and throughput for baseline
                generation_time = time.time() - start_time
                generation_time_str = f"{generation_time:.2f}s"
                
                # Calculate throughput for baseline
                response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
                num_tokens = len(response_tokens)
                throughput = num_tokens / generation_time if generation_time > 0 else 0
                throughput_str = f"{throughput:.2f} tokens/s"
                
                # Start timing for cache version
                cache_start_time = time.time()
                cache_vis_states, cache_response_text = generate_response_with_visualization_cache_and_parallel(
                    model, tokenizer, device, 
                    messages, 
                    gen_length=gen_length, 
                    steps=steps,
                    constraints=parsed_constraints,
                    temperature=temperature,
                    block_length=block_length,
                    remasking=remasking,
                    threshold=threshold
                )
                cache_generation_time = time.time() - cache_start_time
                cache_generation_time_str = f"{cache_generation_time:.2f}s"
                cache_response_tokens = tokenizer.encode(cache_response_text, add_special_tokens=False)
                cache_num_tokens = len(cache_response_tokens)
                cache_throughput = cache_num_tokens / cache_generation_time if cache_generation_time > 0 else 0
                cache_throughput_str = f"{cache_throughput:.2f} tokens/s"
                
                # Update both histories with their respective responses
                history_baseline[-1][1] = response_text
                history_cache[-1][1] = cache_response_text
                
                # Return the initial state immediately
                yield history_baseline, history_cache, vis_states[0], cache_vis_states[0], response_text, generation_time_str, throughput_str, cache_generation_time_str, cache_throughput_str
                
                # Then animate through visualization states
                for state in vis_states[1:]:
                    time.sleep(delay)
                    yield history_baseline, history_cache, state, cache_vis_states[0], response_text, generation_time_str, throughput_str, cache_generation_time_str, cache_throughput_str
                
                for state in cache_vis_states[1:]:
                    time.sleep(delay)
                    yield history_baseline, history_cache, vis_states[-1], state, response_text, generation_time_str, throughput_str, cache_generation_time_str, cache_throughput_str
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(error_msg)
                
                # Show error in visualization
                error_vis = [(error_msg, "red")]
                
                # Don't update histories with error
                yield history_baseline, history_cache, error_vis, error_vis, error_msg, "0.00s", "0.00 tokens/s", "0.00s", "0.00 tokens/s"
        
        def clear_conversation():
            """Clear the conversation history"""
            empty_history = []
            empty_response = ""
            empty_vis = []
            time_str = "0.00s"
            throughput_str = "0.00 tokens/s"
            
            return (
                empty_history,  # chat_history_baseline
                empty_history,  # chat_history_cache
                empty_history,  # chatbot_ui
                empty_history,  # chatbot_ui_copy
                empty_response,  # current_response
                empty_vis,      # output_vis
                time_str,       # generation_time
                throughput_str, # throughput
                empty_vis,      # output_vis_copy
                time_str,       # generation_time_copy
                throughput_str  # throughput_copy
            )
        
        # EVENT HANDLERS
        
        # Clear button handler
        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=[chat_history_baseline, chat_history_cache, chatbot_ui, chatbot_ui_copy, current_response, output_vis, generation_time, throughput, output_vis_copy, generation_time_copy, throughput_copy]
        )
        
        # User message submission flow (2-step process)
        # Step 1: Add user message to history and update UI
        msg_submit = user_input.submit(
            fn=user_message_submitted,
            inputs=[user_input, chat_history_baseline, chat_history_cache, gen_length, steps, constraints_input, visualization_delay],
            outputs=[chat_history_baseline, chat_history_cache, chatbot_ui, chatbot_ui_copy, user_input, output_vis, output_vis_copy, current_response, generation_time, throughput, generation_time_copy, throughput_copy]
        )
        
        # Also connect the send button
        send_click = send_btn.click(
            fn=user_message_submitted,
            inputs=[user_input, chat_history_baseline, chat_history_cache, gen_length, steps, constraints_input, visualization_delay],
            outputs=[chat_history_baseline, chat_history_cache, chatbot_ui, chatbot_ui_copy, user_input, output_vis, output_vis_copy, current_response, generation_time, throughput, generation_time_copy, throughput_copy]
        )
        
        # Step 2: Generate bot response
        # This happens after the user message is displayed
        msg_submit.then(
            fn=bot_response,
            inputs=[
                chat_history_baseline, chat_history_cache, gen_length, steps, constraints_input, 
                visualization_delay, temperature, block_length,
                remasking_strategy, threshold
            ],
            outputs=[chatbot_ui, chatbot_ui_copy, output_vis, output_vis_copy, current_response, generation_time, throughput, generation_time_copy, throughput_copy]
        )
        
        send_click.then(
            fn=bot_response,
            inputs=[
                chat_history_baseline, chat_history_cache, gen_length, steps, constraints_input, 
                visualization_delay, temperature, block_length,
                remasking_strategy, threshold
            ],
            outputs=[chatbot_ui, chatbot_ui_copy, output_vis, output_vis_copy, current_response, generation_time, throughput, generation_time_copy, throughput_copy]
        )
        
    return demo


if __name__ == "__main__":
    prompt = "Jen and Tyler are gymnasts practicing flips. Jen is practicing the triple-flip while Tyler is practicing the double-flip. Jen did sixteen triple-flips during practice. Tyler flipped in the air half the number of times Jen did. How many double-flips did Tyler do?"

    prompt2 = """
You are a powerful diffusion-based language model (dLLM) trained to iteratively refine text through a denoising process. 
Your goal is to produce a coherent, creative, and logically consistent piece of writing over multiple denoising steps. 
At each step, you will receive a noisy or partially masked version of the text, and your job is to fill in or improve the missing or low-confidence tokens 
so that the overall content becomes more complete, fluent, and stylistically appropriate.

Guidelines for Refinement:

1. **Semantic Coherence**
   - Always maintain the core meaning and topic from previous steps.
   - Use context to predict missing information logically.
   - Avoid introducing contradictions between earlier and later parts of the text.

2. **Stylistic Consistency**
   - Match tone and style across all sentences.
   - If the genre is unclear, prefer a neutral, academic, and well-structured style.
   - Use natural transitions between paragraphs and ideas.

3. **Iterative Improvement**
   - When filling in noise or masked regions, consider global sentence structure before generating local tokens.
   - Try to smooth grammar and rhythm while keeping semantic precision.
   - Minor stylistic changes are acceptable if they improve readability.

4. **Handling Ambiguity**
   - If the prompt or partial text contains uncertainty, propose the most likely completion first.
   - When multiple valid options exist, favor those that align with the overall context or tone.

5. **Long-Range Dependencies**
   - Pay attention to consistency in characters, events, or argument structure across distant tokens.
   - Avoid abrupt topic shifts between denoising iterations.

6. **Fidelity to Conditioning**
   - If a conditioning input (like an outline, question, or target topic) is provided, 
     ensure every refinement step moves closer to fulfilling it.

7. **Controlled Creativity**
   - Encourage expressive phrasing and subtle variation, but ensure clarity and purpose.
   - Avoid unnecessary repetition or overuse of adjectives.

8. **Noise-Aware Behavior**
   - In early diffusion steps, prioritize reconstructing grammatical structure and rough semantics.
   - In later steps, focus on fine-grained word choice, fluency, and stylistic polish.

---

Example Conditioning Context:

Input Topic: "The evolution of human–AI collaboration in scientific research."

Your task is to generate a 4–6 paragraph essay that gradually evolves from abstract ideas 
to specific applications and implications, showing clear reasoning and factual grounding. 
In early steps, focus on structural coherence (introduction, body, conclusion). 
In later steps, refine lexical choice, transitions, and flow.

At each diffusion step, you will:
- Receive a noisy or masked version of the current text.
- Predict the denoised version with higher fluency and clarity.
- Output the refined text as your denoised result.

---

Final Output Expectations:
- Smooth, logically structured essay with clear topic sentences.
- Cohesive argument flow.
- Natural human-like phrasing.
- No residual masking tokens or artifacts.

[End of Prompt]

"""

    # _, final_text = generate_response_with_visualization(
    for _ in range(10):

        _, final_text = generate_response_with_visualization(
            model, tokenizer, device, 
            messages=[{"role": "user", "content": prompt}], 
            gen_length=64,
            steps=64,
        )
        _, final_text = generate_response_with_visualization_cache_and_parallel(
            model, tokenizer, device, 
            messages=[{"role": "user", "content": prompt}], 
            gen_length=64,
            steps=64,
        )

    print("Final response:", final_text)


# # Launch the demo
# if __name__ == "__main__":
#     print("Creating the demo...")
#     demo = create_chatbot_demo()

#     print("Launching the demo...")
#     demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7777)