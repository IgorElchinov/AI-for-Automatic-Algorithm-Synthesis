from agent.models import OpenRouterClient

c = OpenRouterClient(model='cohere/north-mini-code:free')
resp = c.generate('Say hello and return only "OK"')
print('TEXT:', resp.text)
print('RAW MODEL:', resp.raw.get('model'))
