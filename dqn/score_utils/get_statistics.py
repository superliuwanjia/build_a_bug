import sys
import matplotlib.pyplot as plt
import scipy.signal
import os
            # 'type:Episode gameNum:11526 score:25 framesInGame:3128 accumFrames:12874186

def accum_helper(sum_vals, max_vals, min_vals, list_append, val):
    max_vals = max(max_vals, val)
    list_append.append(val)
    return sum_vals + val, max(max_vals, val), min(min_vals, val)

def get_statistics(filename=None):
    filename = filename if filename else sys.argv[1]
    print(filename)
    with open(filename, 'r') as f:
        content = f.readlines()
        # content = content[:61057]
        counts = [30, 50, 100, 200]
        first_lines = content[:counts[-1]]
        last_lines = reversed(content[~counts[-1]:-1])
        first_scores = []
        last_scores = []
        max_first, max_last = 0, 0
        min_first, min_last = float('inf'), float('inf')
        sum_first, sum_last = 0, 0
        lines = 0
        milestone = 0

        for fir, l in zip(first_lines, last_lines):
            val = int(fir.split()[2].split(":")[-1])
            sum_first, max_first, min_first = accum_helper(sum_first, max_first, min_first, first_scores, val)

            val = int(l.split()[2].split(":")[-1])
            sum_last, max_last, min_last = accum_helper(sum_last, max_last, min_last, last_scores, val)

            if lines == counts[milestone] - 1:
                count = counts[milestone] - 1
                print 'Score statistics for the first {} episodes: Mean: {}, Max: {}, Min: {}'.format(count,
                                                                                                sum_first / float(count),
                                                                                                max_first,
                                                                                                min_first)
                print 'Score statistics for the last {} episodes: Mean: {}, Max: {}, Min: {}'.format(count,
                                                                                                sum_last / float(count),
                                                                                                max_last,
                                                                                                min_last)
                print
                milestone += 1
            lines += 1
        print 'Total Episodes: {}'.format(len(content))
        accum_frames = []
        game_nums = []
        scores = []
        for x in content[:-1]:
            stripped = x.strip()
            split = stripped.split(' ')
            # print split
            accum_frames.append(int(split[4].split(":")[-1]))
            game_nums.append(int(split[1].split(":")[-1]))
            scores.append(int(split[2].split(":")[-1]))

        print 'Total Frames seen: {:,}'.format(int(content[-2].split(' ')[-1].split(":")[-1]))
        # plt.plot(accum_frames, scores)
        # plt.savefig(sys.argv[1] + 'plot_train.png')
        # plt.show()
        return game_nums, scores


def show_plots_pergame():
    # games = [x[0] for x in os.walk(sys.argv[1])][1:]
    games = ['data/qbert/qbert-ema', 'data/qbert/qbert-None', 'data/qbert/qbert-stack']
    print(games)

    # games = os.listdir(sys.argv[1])
    # games = [os.path.abspath(g) for g in games]
    # print games
    # games = [g for g in games if os.path.isdir(g)]
    print games
    all_games = []
    all_scores = []
    min_game_count = float('inf')
    for game in games:

        train_path = game + '/train.txt'
        g, s = get_statistics(train_path)
        print(len(g), len(s))
        all_scores.append(s)
        all_games.append(g)
        # if len(g) < min_game_count:
        #     all_games = g
        #     min_game_count = len(g)

    min_len = len(all_games)
    # print(len(all_scores))
    print('min length - {}'.format(min_len))
    # plt.plot(all_games, scipy.signal.savgol_filter(all_scores[0][:min_len], 551, 6), '-r',
    #          all_games, scipy.signal.savgol_filter(all_scores[1][:min_len], 551, 6), 'b',
    #          all_games, scipy.signal.savgol_filter(all_scores[2][:min_len], 551, 6), 'g')
    labels = [g.split('/')[-1] for g in games]
    plots = []

    for i, s in enumerate(all_scores):
        print(labels[i])
        plots.append(plt.plot(all_games[i], scipy.signal.savgol_filter(s, 551, 6), label=labels[i]))
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=2, mode="expand", borderaxespad=0.)
    # Place a legend to the right of this smaller subplot.
    # plt.legend()
    # plt.legend(plots, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(sys.argv[1] + 'plot_train.png')
    plt.show()
if __name__ == '__main__':
    # get_statistics()
    show_plots_pergame()
