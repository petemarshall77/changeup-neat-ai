import random
import neat
import os
import time

class Goal():

    def __init__(self):
        self.bottom = '.'
        self.middle = '.'
        self.top    = '.'

    def score(self, color):
        if self.top != '.':
            return False

        if self.bottom == '.':
            self.bottom = color
        elif self.middle == '.':
            self.middle = color
        else:
            self.top = color

        return True

    def de_score(self):
        if self.bottom == ',':
            return False

        self.bottom = self.middle
        self.middle = self.top
        self.top = '-'

        return True

    def owned_by(self):
        """ Who owns the goal, i.e. has the topmost ball """
        if self.top != '.':
            return self.top
        elif self.middle != '.':
            return self.middle
        else:
            return self.bottom

    def descriptor(self):
        """ Return a number that describes the goal state for NN purposes """
        descriptor = 0
        if self.bottom == 'R': descriptor += 2
        if self.bottom == 'B': descriptor += 1
        if self.middle == 'R': descriptor += 20
        if self.middle == 'B': descriptor += 10
        if self.top    == 'R': descriptor += 200
        if self.top    == 'B': descriptor += 100
        return float(descriptor/1000.0)


    def get_score(self):
        """ Get the score for the goal, return a tuple (<red>, <blue>) """
        scores = [0, 0]

        # One point for each blaa in goal
        for index, color in enumerate(['R', 'B']):
            if self.bottom == color: scores[index] += 1
            if self.middle == color: scores[index] += 1
            if self.top == color: scores[index] += 1
        return (scores[0], scores[1])


class Field():
    def __init__(self):
        self.goals = []
        for i in range(9):
            self.goals.append(Goal())


    def score(self, goal, color):
        return self.goals[goal].score(color)

    def get_score(self):
        """ get the score for the field, return a tuple (<red>, <blue>) """
        scores = [0,0] # red, blue

        # 1 point for each blaa in goal
        for goal in self.goals:
            goal_score = goal.get_score()
            scores[0] += goal_score[0]
            scores[1] += goal_score[1]

        # Six points for each row owned
        rows = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for index, color in enumerate(['R', 'B']):
            for row in rows:
                if (self.goals[row[0]].owned_by() == color) and (self.goals[row[1]].owned_by() == color) and (self.goals[row[2]].owned_by() == color):
                    scores[index] += 6

        return (scores[0], scores[1])

    def print(self):
        print("+-----------------------+")
        for i in range(3):
            print("|%s          %s          %s|" % (self.goals[i*3].top, self.goals[i*3+1].top, self.goals[i*3+2].top))
            print("|%s          %s          %s|" % (self.goals[i*3].middle, self.goals[i*3+1].middle, self.goals[i*3+2].middle))
            print("|%s          %s          %s|" % (self.goals[i*3].bottom, self.goals[i*3+1].bottom, self.goals[i*3+2].bottom))
            if i < 2:
                print("|                       |")

        print("+-----------------------+")

        scores = self.get_score()
        print("Red = %02d        Blue = %02d" % scores)

        print()

    def get_descriptors(self):
        descriptors = []
        for i in range(9):
            descriptors.append(self.goals[i].descriptor())
        return descriptors


class Player():

    def __init__(self, name, strategy):
        self.name = name
        self.strategy = strategy

    def add_net(self, net):
        self.net = net

    def make_move(self, field, color):
        if self.strategy == 'random':
            return self.random_choice(field, color)
        elif self.strategy == 'neat-ai':
            return self.neat_choice(field, color)

    def random_choice(self, field, color):
        """ Really simple - simply add ball to random goal """
        field.goals[random.randint(0,8)].score(color)

    def neat_choice(self, field, color):
        if color == 'R':
            input_list = [1.0]
        else:
            input_list = [-1.0]

        descriptors = field.get_descriptors()
        for descriptor in descriptors:
            input_list.append(descriptor)

        nn_inputs = tuple(input_list)
        output = self.net.activate(nn_inputs)
        max_output_idx = output.index(max(output))

        if max_output_idx < 9:
            field.goals[max_output_idx].score(color)
        else:
            field.goals[max_output_idx-9].de_score()


def play_a_game(field, player1, player2, showmoves=False):

    for i in range(10):
        player1.make_move(field, 'R')
        player2.make_move(field, 'B')

    if showmoves == True:
        field.print()

def play_a_tournament(players, rounds=1):

    high_score = 0
    total_score = 0
    wins = [0]*len(players)
    for _ in range(rounds):
        for i in range(len(players)):
            for j in range(len(players)):
                if i != j:
                    field = Field()
                    play_a_game(field, players[i], players[j])
                    score = field.get_score()
                    total_score += score[0] + score[1]
                    if score[0] > score[1]:
                        wins[i] += 1
                        if score[0] > high_score:
                            high_score = score[0]
                    elif score[1] > score[0]:
                        wins[j] += 1
                        if score[1] > high_score:
                            high_score = score[1]
                    #print("Game", i, j, players[i].name, score[0], "-", players[j].name, score[1])

    return wins, high_score, total_score


def eval_genomes(genomes, config):
    # Create the genome itself
    nets = []
    players = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        player = Player("NEAT-%s" % genome_id, 'neat-ai')
        player.add_net(net)
        players.append(player)
        ge.append(genome)

    wins, high_score, total_score = play_a_tournament(players)

    for i in range(len(wins)):
        #print("Player", i, players[i].name, wins[i], "wins.")
        ge[i].fitness = wins[i]
    print("High score:", high_score)
    print("Average score:", total_score/(len(wins)*99))
    time.sleep(10)


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create The Population, Which Is The Top-Level Object For A Neat Run.
    p = neat.Population(config)

    # Add A Stdout Reporter To Show Progress In The Terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #P.Add_reporter(Neat.Checkpointer(5))

    # Run For Up To 50 Generations.
    winner = p.run(eval_genomes, 50)

    # Show Final Stats
    #print('\nbest Genome:\n{!s}'.format(winner))


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_file)




player1 = Player('Foo', 'random')
player2 = Player('Bar', 'random')
player3 = Player('Baz', 'random')
player4 = Player('Quux', 'random')

players = [player1, player2, player3, player4]
play_a_tournament(players)
