from transformers import BertTokenizer, BertForMaskedLM, BertConfig
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from dataloader import DynamicMaskingDataset

from torch.utils.tensorboard import SummaryWriter

import random
from tqdm import tqdm
import os
from datasets import load_dataset

# set cuda device
torch.cuda.set_device('cuda:7')

# load config and tokenizer
config = BertConfig.from_pretrained('prajjwal1/bert-mini')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# max position length 수정(512 => 64)
# padding이 크면 masking에 대한 예측이 쉬울 수 있음(예상)
config.max_position_embeddings = 64

# load datasets
dataset = load_dataset('wikitext', 'wikitext-103-v1')

train_dataset = DynamicMaskingDataset(config, dataset['train'], tokenizer, config.max_position_embeddings)
valid_dataset = DynamicMaskingDataset(config, dataset['validation'], tokenizer, config.max_position_embeddings)
test_dataset = DynamicMaskingDataset(config, dataset['test'], tokenizer, config.max_position_embeddings)


def main(savename):
    epochs = 40
    batch_size = 256
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = BertForMaskedLM(config=config)
    optimizer = transformers.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 10000,
                                                             len(train_dataloader) * epochs)

    if torch.cuda.device_count() > 1:
        device = 'cuda'
        model = torch.nn.DataParallel(model, device_ids = [7,0,1,2,3,4,5,6], output_device=7)
    elif torch.cuda.device_count() == 1:
        device = 'cuda'

    else:
        device = 'cpu'

    # tensorboard
    writer = SummaryWriter(f'runs/{savename}')

    # training
    for epoch in tqdm(range(epochs)):
        running_loss = 0
        for i, batch in enumerate(train_dataloader):
            model.train()
            model.to(device)
            model.zero_grad()
            input_ids, token_type_ids, attention_mask, labels = (t.to(device) for t in batch.values())
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            labels=labels)

            # backward
            loss = outputs[0]
            loss.mean().backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.mean().item()

            if i % 1000 == 999:  # 매 1000 step마다 기록
                # loss 기록
                writer.add_scalar('Train_Loss', running_loss / 1000, epoch * len(train_dataloader) + i)
                # accuracy 계산 및 기록
                logits = outputs[1]
                logits = logits.argmax(dim=-1)
                accuracy = (logits.cpu().numpy() == labels.cpu().numpy()).mean()
                writer.add_scalar('MaskedLM_Train_Acc', accuracy, epoch * len(train_dataloader) + i)

                # learning rate 기록
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch * len(train_dataloader) + i)

                print(f'[epochs] - {epoch} ======= [steps] - {epoch * len(train_dataloader) + i}')
                print(f'[train_loss] - {running_loss / 1000} ========= Masked LM accuracy - {accuracy}')

                running_loss = 0

            if i % 1000 == 999: # 매 1000 step마다 validation loss 기록 및 모델 저장
                eval_loss = 0
                eval_accu = 0
                with torch.no_grad():
                    model.eval()
                    model.zero_grad()
                    for batch in valid_dataloader:
                        input_ids, token_type_ids, attention_mask, labels = (t.to(device) for t in batch.values())
                        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                        attention_mask=attention_mask,
                                        labels=labels)


                        loss = outputs[0].mean().item()
                        eval_loss += loss

                        eval_logits = outputs[1].argmax(dim=-1)
                        eval_accu += (eval_logits.cpu().numpy() == labels.cpu().numpy()).mean()

                    writer.add_scalar('Eval_Loss', eval_loss / len(valid_dataloader), epoch * len(train_dataloader) + i)
                    writer.add_scalar('MaskedLM_Eval_Acc', eval_accu / len(valid_dataloader), epoch * len(train_dataloader) + i)

                    print(f'[epochs] - {epoch} ======= [steps] - {epoch * len(train_dataloader) + i}')
                    print(f'[eval_loss] - {eval_loss / len(valid_dataloader)} ========= Masked LM Eval accuracy - {eval_accu / len(valid_dataloader)}')

                # save model
                model.train()
                path = f'./BERT_pretrained/{savename}/epoch_{epoch}_step_{epoch * len(train_dataloader) + i}'
                os.makedirs(path, exist_ok=True)
                torch.save(model.state_dict(), f'{path}/model.pt')
                torch.save(optimizer.state_dict(), f'{path}/optimizer.pt')
                torch.save(scheduler.state_dict(), f'{path}/scheduler.pt')


    writer.close()


if __name__ == '__main__':
    main('Original_BERTmini')