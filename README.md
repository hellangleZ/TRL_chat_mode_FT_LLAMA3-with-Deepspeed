# TRL_chat_mode_FT_LLAMA3


Support Deepspeed to reduce RAM


![image](https://github.com/hellangleZ/TRL_chat_mode_FT_LLAMA3/assets/15274284/4fc30594-e6f5-4ad6-a580-b4b2b5ef93c7)


![image](https://github.com/hellangleZ/TRL_chat_mode_FT_LLAMA3/assets/15274284/89a8796c-55b5-4d1c-9093-109cc6e7dae1)

how to run:

accelerate launch --config_file=dp_z2.yaml --gradient_accumulation_steps 4 trl_ft_chatmode.py
