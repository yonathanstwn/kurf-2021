from additional import visualize


def create_schedule(path):
    new_schedule = {'schedule': {'agent0': []}}
    for i in range(len(path)):
        new_schedule['schedule']['agent0'].append({'x': path[i][0], 'y': path[i][1], 't': i})
    return new_schedule


def create_new_dct(new_graph, path, old_dct):
    new_dct = {'agents': [{'goal': list(path[-1]), 'name': 'agent0',
                           'start': list(path[0])}],
               'map': {'dimensions': old_dct['map']['dimensions'], 'obstacles': []}}
    # Obstacles
    obstacles = [node for node in new_graph.nodes if new_graph.nodes[node].get('area_type') == 1]
    print(obstacles)
    new_dct['map']['obstacles'] = obstacles
    # Return
    return new_dct


def animate(new_graph, path, old_dct):
    new_dct = create_new_dct(new_graph, path, old_dct)
    new_schedule = create_schedule(path)
    animation = visualize.Animation(new_dct, new_schedule)
    animation.show()
