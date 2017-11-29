import sys
import matplotlib.pyplot as plt
            # 'type:Episode gameNum:11526 score:25 framesInGame:3128 accumFrames:12874186

def accum_helper(sum_vals, max_vals, min_vals, list_append, val):
    max_vals = max(max_vals, val)
    list_append.append(val)
    return sum_vals + val, max(max_vals, val), min(min_vals, val)

def get_statistics():
    with open(sys.argv[1], 'r') as f:
        content = f.readlines()

        count = 2
        first_lines = content[:count]
        last_lines = content[~count:]
        first_scores = []
        last_scores = []
        max_first, max_last = 0, 0
        min_first, min_last = float('inf'), float('inf')
        sum_first, sum_last = 0, 0
        for f, l in zip(first_lines, last_lines):
            val = int(f.split()[2].split(":")[-1])
            sum_first, max_first, min_first = accum_helper(sum_first, max_first, min_first, first_scores, val)

            val = int(l.split()[2].split(":")[-1])
            sum_last, max_last, min_last = accum_helper(sum_last, max_last, min_last, last_scores, val)

        print 'Statistics for the first {} episodes: Mean: {}, Max: {}, Min: {}'.format(count,
                                                                                        sum_first / float(count),
                                                                                        max_first,
                                                                                        min_first)
        print 'Statistics for the last {} episodes: Mean: {}, Max: {}, Min: {}'.format(count,
                                                                                        sum_last / float(count),
                                                                                        max_last,
                                                                                        min_last)
        accum_frames = []
        game_nums = []
        scores = []
        for x in content:
            stripped = x.strip()
            split = stripped.split(' ')
            accum_frames.append(int(split[4].split(":")[-1]))
            game_nums.append(int(split[1].split(":")[-1]))
            scores.append(int(split[2].split(":")[-1]))
        print accum_frames, scores
        plt.plot(accum_frames, scores)
        # plt.show()
if __name__ == '__main__':
    get_statistics()
