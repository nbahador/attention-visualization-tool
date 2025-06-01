# Attention Matrix Input Format

## File Naming Convention
Attention matrices should follow this naming pattern:
`attention_epoch_{epoch}_sample_{sample_id}_L{layer}H{head}.csv`

Example:
`attention_epoch_15.0_sample_5388_L0H0.csv`

## File Format
- CSV format with header row
- Square matrix where each row represents attention from one token to all other tokens
- Values should be between 0 and 1 (attention weights)

