import torch
from transformers import DistilBertModel, BertModel, BertConfig

class DistilBertForWhQuestionInference(torch.nn.Module):
	def __init__(self):
		super(DistilBertForWhQuestionInference, self).__init__()
		self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
		self.pre_classifier = torch.nn.Linear(768, 768)
		self.dropout = torch.nn.Dropout(0.3)
		self.classifier = torch.nn.Linear(768, 4)
		self.softmax = torch.nn.Softmax(dim=1)

	def forward(self, input_ids, attention_mask):
		output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
		hidden_state = output_1[0]
		pooler = hidden_state[:, 0]
		pooler = self.pre_classifier(pooler)
		pooler = torch.nn.ReLU()(pooler)
		pooler = self.dropout(pooler)
		output = self.classifier(pooler)
		output = self.softmax(output)
		return output

class BertForWhQuestionInference(torch.nn.Module):
	def __init__(self, pretrained_path):
		super(BertForWhQuestionInference, self).__init__()
		if pretrained_path is None:
			self.l1 = BertModel.from_pretrained('bert-base-uncased')
		else:
			self.l1 = BertModel.from_pretrained(pretrained_path)
		self.pre_classifier = torch.nn.Linear(768, 768)
		self.dropout = torch.nn.Dropout(0.3)
		self.classifier = torch.nn.Linear(768, 4)
		self.softmax = torch.nn.Softmax(dim=1)

	def forward(self, input_ids, attention_mask):
		output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
		hidden_state = output_1[0]
		pooler = hidden_state[:, 0]
		pooler = self.pre_classifier(pooler)
		pooler = torch.nn.ReLU()(pooler)
		pooler = self.dropout(pooler)
		output = self.classifier(pooler)
		output = self.softmax(output)
		return output