import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) > 3 or  len(result) <= 3 # alterado p/ aceitar 3 métricas adicionais, alterar se acrescentar
        #print(self.results)
        #print(run)
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result[:6])

    def print_statistics(self, run=None):
        if run is not None:
            # melhores resultados para cada run
            result = 100 * torch.tensor(self.results[run])
            #argmax = result[:, 1].argmax().item() # result[:, 1] vai à 2a coluna, dos scores de validation, buscar o idx do mais elevado
            argmax = result[:, 0].argmax().item()
            if len(result[argmax,]) < 3:
                print(f'Run {run + 1:02d}:')
                print(f'Final Test: {result[:, 0].max():.2f}')
                print(f'Final Acc: {result[:, 1].max():.2f}')
                #print(f'  Final Test: {result[argmax, 0]:.2f}')
                #print(f'   Final Acc: {result[argmax, 1]:.2f}')
            else:
                print(f'Run {run + 1:02d}:')
                print(f'Highest Train: {result[:, 0].max():.2f}')
                print(f'Highest Valid: {result[:, 1].max():.2f}')
                print(f'  Final Train: {result[argmax, 0]:.2f}') 
                print(f'   Final Test: {result[argmax, 2]:.2f}')  # teste final tem em conta o melhor validation score
            # acrescentado
            if len(result[argmax,]) > 3:
                print(f'  Highest Acc: {result[:, 3].max():.2f}') # para ter ideia do Roc-Auc mais elevado obtido
                print(f'  Final Test Acc: {result[argmax, 3]:.2f}') # rocauc final a ter em conta melhor validation score
                #print(f'  Highest Recall: {result[:, 4].max():.2f}')
                #print(f'  Final Recall: {result[argmax, 4]:.2f}')
                #print(f'  Highest Precision: {result[:, 5].max():.2f}')
                #print(f'  Final Precision: {result[argmax, 5]:.2f}')
        else:
            #print(self.results)
            result = 100 * torch.tensor(self.results)
            # melhores resultados de todas as runs
            best_results = []
            for r in result:
                if len(r[0]) < 3:
                    train1 = r[:, 0].max().item()
                    valid = r[:, 1].max().item()
                    #train2 = r[r[:, 1].argmax(), 0].item()
                    #test = r[r[:, 1].argmax(), 1].item()
                    best_results.append((train1, valid))
                else:    
                    train1 = r[:, 0].max().item()
                    valid = r[:, 1].max().item()
                    train2 = r[r[:, 1].argmax(), 0].item()
                    test = r[r[:, 1].argmax(), 2].item()
                # acrescentado
                if len(r[0]) > 3:
                    acc = r[r[:, 1].argmax(), 3].item()
                    #recall = r[r[:, 1].argmax(), 4].item()
                    #precision = r[r[:, 1].argmax(), 5].item()
                    #best_results.append((train1, valid, train2, test, acc, recall, precision))
                    best_results.append((train1, valid, train2, test, acc))
                #else:
                    #best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)
            print(best_result)
            print(f'All runs:')
            if len(best_result[0, :]) <= 4:
                r = best_result[:, 0]
                print(f'Highest Test: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, 1]
                print(f'Highest Acc: {r.mean():.2f} ± {r.std():.2f}')
                #r = best_result[:, 2]
                #print(f'  Final Test: {r.mean():.2f} ± {r.std():.2f}')
                #r = best_result[:, 3]
                #print(f'   Final Acc: {r.mean():.2f} ± {r.std():.2f}')

            else:
                r = best_result[:, 0]
                print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, 1]
                print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, 2]
                print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, 3]
                print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            # acrescentado
            if len(best_result[0, :]) > 4:
                r = best_result[:, 4]
                print(f'   Final Acc: {r.mean():.2f} ± {r.std():.2f}')
                #r = best_result[:, 5]
                #print(f'   Final Recall: {r.mean():.2f} ± {r.std():.2f}')
                #r = best_result[:, 6]
                #print(f'   Final Precision: {r.mean():.2f} ± {r.std():.2f}')
