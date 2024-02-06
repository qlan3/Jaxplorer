import os
import sys
import argparse
from slack_sdk.webhook import WebhookClient


def main(argv):
  parser = argparse.ArgumentParser(description="Slack messenger")
  parser.add_argument('--host', type=str, default='', help='the host')
  parser.add_argument('--msg', type=str, default='炼丹完毕！请主人检查仙丹质量😃', help='the message')
  args = parser.parse_args()
  
  url = 'https://hooks.slack.com/services/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' # Personal url, do not leak out
  webhook = WebhookClient(url)
  if len(args.host) == 0:
    host = os.uname()[1]
  else:
    host = args.host
  response = webhook.send(text=f'{host}: {args.msg}')
  assert response.status_code == 200
  assert response.body == "ok"

if __name__=='__main__':
  main(sys.argv)