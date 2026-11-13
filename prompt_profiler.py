import sys, json, argparse

class TokenProfiler:
    def __init__(self, cost_per_1k=0.002):
        self.cost_per_1k = cost_per_1k
    def estimate(self, text):
        tokens = len(text.split()) * 1.33
        cost = (tokens / 1000) * self.cost_per_1k
        return {"tokens": int(tokens), "cost": round(cost, 6)}

def main():
    parser = argparse.ArgumentParser(description="LLM Cost Profiler")
    parser.add_argument("--text", help="Input prompt text")
    args = parser.parse_args()
    if args.text:
        profiler = TokenProfiler()
        print(json.dumps(profiler.estimate(args.text)))
    else:
        print("Usage: python prompt_profiler.py --text 'your prompt here'")

if __name__ == '__main__':
    main()