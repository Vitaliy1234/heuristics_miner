import pandas as pd


LOG_FILE = 'logs.csv'
CASE_ID = 'case:concept:name'
TIMESTAMP = 'time:timestamp'
ACTIVITY_KEY = 'concept:name'
MIN_ACT_COUNT = 1
MIN_DFG_OCCURRENCES = 1


def apply_hue_miner(log_df,
                    dependency_thresh=0.65,
                    and_measure_thresh=0.65,
                    dfg_pre_cleaning_noise_thresh=0.05):
    """Discover process logs with Heuristic Miner algorithm:

        Function returns DFG represented by:
            - list of nodes (str)
            - dict of edges with (startnode, endnode) as keys and dependency values as a value

        Args:
            log_df (pandas.DataFrame): DataFrame with event logs, which contains columns:
                'case:concept:name':  column with unique id of an event trace.
                'concept:name': column with name of an activity in an event trace.
                'time:timestamp': column with Timestamp of an activity in a particular trace.
            dependency_thresh (float): dependency threshold value [0.0, 1.0], default=0.65
            and_measure_thresh (float): and dependency threshold value [0.0, 1.0], default=0.65
            dfg_pre_cleaning_noise_thresh (float): precleaning dependency threshold value [0.0, 1.0],
                default=0.05


        Returns:
            A dict mapping keys to the corresponding DFG with nodes and edges. For example:
            {
                'nodes': ['startevent', 'endevent'],  # list of nodes (str)
                'edges': {('startevent', 'endevent'): 1.0}  # dict of edges, {(tuple of str): (float)}
            }

        """
    nodes = []

    loops_length_two_thresh = dependency_thresh
    dfg_freq = make_dfg_graph(log_df)

    activity_triples = make_activity_triples(log_df)
    cleaned_dfg = clean_dfg_from_noise(log_df, dfg_freq, dfg_pre_cleaning_noise_thresh)
    dfg = {}
    result_dfg = {}
    freq_triples_matrix = {}
    loop_length_two = {}

    # делаем матрицу циклов длины 2
    for el in activity_triples:
        act_1, act_2, act_3 = el
        value = activity_triples[el]

        if act_1 == act_3 and act_1 != act_2:
            if act_1 not in freq_triples_matrix:
                freq_triples_matrix[act_1] = {}
            freq_triples_matrix[act_1][act_2] = value

    # считаем зависимости
    for el in cleaned_dfg:
        act_1, act_2 = el
        value = cleaned_dfg[el]

        if act_1 != act_2:
            inv_couple = (act_2, act_1)
            c_1 = value
            if inv_couple in cleaned_dfg:
                c_2 = cleaned_dfg[inv_couple]
                dep = (c_1 - c_2) / (c_1 + c_2 + 1)
            else:
                dep = c_1 / (c_1 + 1)
        else:
            dep = value / (value + 1)

        dfg[el] = dep

    activities = log_df[ACTIVITY_KEY].unique()
    activities_occurrences = {}
    for act in activities:
        activities_occurrences[act] = sum_activities_count(cleaned_dfg, [act])

    # составляем итоговый граф
    for elem in dfg:
        n_1, n_2 = elem
        condition1 = n_1 in activities_occurrences and activities_occurrences[n_1] >= MIN_ACT_COUNT
        condition2 = n_2 in activities_occurrences and activities_occurrences[n_2] >= MIN_ACT_COUNT
        condition3 = cleaned_dfg[elem] >= MIN_DFG_OCCURRENCES
        condition4 = dfg[elem] >= dependency_thresh
        condition = condition1 and condition2 and condition3 and condition4

        if condition:
            if n_1 not in nodes:
                nodes.append(n_1)
            if n_2 not in nodes:
                nodes.append(n_2)

            result_dfg[elem] = dfg[elem]

    for node in nodes:
        loop_length_two[node] = {}
        if node in freq_triples_matrix:
            for node_2 in freq_triples_matrix[node]:
                if (node, node_2) in cleaned_dfg:
                    c_1 = cleaned_dfg[node, node_2]
                else:
                    c_1 = 0
                v_1 = freq_triples_matrix[node][node_2] if node in freq_triples_matrix and node_2 in freq_triples_matrix[
                    node] else 0
                v_2 = freq_triples_matrix[node_2][node] if node_2 in freq_triples_matrix and node in freq_triples_matrix[
                    node_2] else 0
                l2l = (v_1 + v_2) / (v_1 + v_2 + 1)
                if l2l >= loops_length_two_thresh:
                    loop_length_two[node][node_2] = c_1

    added_loops = set(nodes)
    result_result_dfg = {}
    for edge in result_dfg:
        result_result_dfg[edge] = result_dfg[edge]
        for node_2 in loop_length_two[edge[0]]:
            if (edge[0], node_2) in cleaned_dfg \
                    and cleaned_dfg[(edge[0], node_2)] >= MIN_DFG_OCCURRENCES \
                    and edge[0] in activities_occurrences \
                    and activities_occurrences[edge[0]] >= MIN_ACT_COUNT \
                    and node_2 in activities_occurrences \
                    and activities_occurrences[node_2] >= MIN_ACT_COUNT:
                if not (((edge[0], node_2) in dfg and
                         dfg[edge[0], node_2] >= dependency_thresh) or (
                                (node_2, edge[0]) in dfg and
                                dfg[node_2, edge[0]] >= dependency_thresh)):
                    if node_2 not in nodes:
                        nodes.append(node_2)

                    if (edge[0], node_2) not in added_loops:
                        added_loops.add((edge[0], node_2))
                        result_result_dfg[(edge[0], node_2)] = 0.0

                    if (node_2, edge[0]) not in added_loops:
                        added_loops.add((node_2, edge[0]))
                        result_result_dfg[(node_2, edge[0])] = 0.0
    if len(nodes) == 0:
        nodes = activities

    for edge in result_result_dfg:
        print(edge, result_result_dfg[edge])
    print()
    print(nodes)
    return {'nodes': nodes, 'edges': result_result_dfg}


def sum_activities_count(dfg, activities, enable_halving=True):
    """
    Gets the sum of specified attributes count inside a DFG

    Parameters
    -------------
    dfg
        Directly-Follows graph
    activities
        Activities to sum
    enable_halving
        Halves the sum in specific occurrences

    Returns
    -------------
        Sum of start attributes count
    """
    ingoing = get_ingoing_edges(dfg)
    outgoing = get_outgoing_edges(dfg)

    sum_values = 0

    for act in activities:
        if act in outgoing:
            sum_values += sum(outgoing[act].values())
        if act in ingoing:
            sum_values += sum(ingoing[act].values())
        if act in ingoing and act in outgoing:
            sum_values = int(sum_values / 2)

    return sum_values


def clean_dfg_from_noise(log_df, dfg_freq, dfg_pre_cleaning_noise_thresh):
    activities = log_df[ACTIVITY_KEY].unique()
    new_dfg = {}
    activ_max_count = {}

    for act in activities:
        activ_max_count[act] = get_max_activity_count(dfg_freq, act)

    for el in dfg_freq:
        act_1, act_2 = el
        val = dfg_freq[el]

        if val >= min(activ_max_count[act_1] * dfg_pre_cleaning_noise_thresh,
                      activ_max_count[act_2] * dfg_pre_cleaning_noise_thresh):
            new_dfg[el] = dfg_freq[el]

    if not new_dfg:
        return dfg_freq

    return new_dfg


def get_outgoing_edges(dfg):
    """
    Gets outgoing edges of the provided DFG graph
    """
    outgoing = {}
    for el in dfg:
        if not el[0] in outgoing:
            outgoing[el[0]] = {}

        outgoing[el[0]][el[1]] = dfg[el]
    return outgoing


def get_ingoing_edges(dfg):
    """
    Get ingoing edges of the provided DFG graph
    """
    ingoing = {}
    for el in dfg:
        if not el[1] in ingoing:
            ingoing[el[1]] = {}

        ingoing[el[1]][el[0]] = dfg[el]
    return ingoing


def get_max_activity_count(dfg, act):
    """
    Get maximum count of an ingoing/outgoing edge related to an activity

    Parameters
    ------------
    dfg
        Directly-Follows graph
    act
        Activity

    Returns
    ------------
    max_value
        Maximum count of ingoing/outgoing edges to attributes
    """
    ingoing = get_ingoing_edges(dfg)
    outgoing = get_outgoing_edges(dfg)
    max_value_1 = -1
    max_value_2 = -1

    if act in ingoing:
        max_value_1 = max(ingoing[act].values())

    if act in outgoing:
        max_value_2 = max(outgoing[act].values())

    max_value = max(max_value_1, max_value_2)
    return max_value


def make_activity_triples(log_df,
                          case_id_glue=CASE_ID,
                          timestamp_key=TIMESTAMP,
                          window1=-1,
                          window2=-2,
                          activity=ACTIVITY_KEY,
                          ):
    log_df = log_df.sort_values([case_id_glue, timestamp_key])
    log_df_reduced = log_df[[case_id_glue, activity]]
    # сдвигае датафрейм вниз на 1 строку
    log_df_reduced_1 = log_df_reduced.shift(window1)
    # сдвигае датафрейм вниз на 2 строки
    log_df_reduced_2 = log_df_reduced.shift(window2)
    # меняем названия колонок в сдинутых датафреймах
    log_df_reduced_1.columns = [str(col) + '_2' for col in log_df_reduced_1.columns]
    log_df_reduced_2.columns = [str(col) + '_3' for col in log_df_reduced_2.columns]
    df_successive_rows = pd.concat([log_df_reduced, log_df_reduced_1, log_df_reduced_2], axis=1)
    df_successive_rows = df_successive_rows[df_successive_rows[case_id_glue] == df_successive_rows[case_id_glue + '_2']]
    df_successive_rows = df_successive_rows[df_successive_rows[case_id_glue] == df_successive_rows[case_id_glue + '_3']]
    all_columns = set(df_successive_rows.columns)
    all_columns = list(all_columns - {activity, activity + '_2', activity + '_3'})
    directly_follows_grouping = df_successive_rows.groupby([activity, activity + '_2', activity + '_3'])
    directly_follows_grouping = directly_follows_grouping[all_columns[0]]
    freq_triples = directly_follows_grouping.size().to_dict()
    return freq_triples


def make_dfg_graph(log_df,
                   case_id_glue=CASE_ID,
                   timestamp_key=TIMESTAMP,
                   window=-1,
                   activity=ACTIVITY_KEY):
    # сортируем по времени и по id трейсов
    log_df = log_df.sort_values([case_id_glue, timestamp_key])

    log_df_reduced = log_df[[case_id_glue, activity]]
    # сдвигаем все строки вниз на 1, чтоб потом получить пары событий
    df_reduced_shifted = log_df_reduced.shift(-window)
    # переименуем
    df_reduced_shifted.columns = [str(col) + '_2' for col in df_reduced_shifted.columns]
    # соединим сдвинутый и оригинальный датафреймы
    df_successive_rows = pd.concat([log_df_reduced, df_reduced_shifted], axis=1)
    # сделаем так, чтобы пары событий соответствовали одному трейсу
    df_successive_rows = df_successive_rows[df_successive_rows[case_id_glue] == df_successive_rows[case_id_glue + '_2']]

    all_columns = set(df_successive_rows.columns)
    all_columns = list(all_columns - {activity, activity + '_2'})
    directly_follows_grouping = df_successive_rows.groupby([activity + '_2', activity])
    directly_follows_grouping = directly_follows_grouping[all_columns[0]]
    dfg_frequency = directly_follows_grouping.size().to_dict()
    return dfg_frequency


if __name__ == '__main__':
    log = pd.read_csv(LOG_FILE)
    res = apply_hue_miner(log)
