import argparse
from src.train import *
from src.test import *
import sys
torch.autograd.set_detect_anomaly(True)
import shutil
from datetime import datetime
from best_params import BEST_PARAMS_BY_DATASET
from src.data_load.KnowledgeGraph import *
from src.model.controller import Controller


class experiment:
    def __init__(self, args):
        self.args = args

        '''1. prepare data file path, model saving path and log path'''
        self.prepare()

        '''2. load data'''
        self.kg = KnowledgeGraph(args)

        '''3. create model and optimizer'''
        self.model, self.optimizer = self._create_model()
        self.start_epoch = 0

        if self.args.load_checkpoint is not None:
            self.start_epoch = self.load_checkpoint(os.path.join(self.args.load_checkpoint, 'model_best.tar'))
            self.model.args = self.args
            self.model.kg = self.kg
        self.args.logger.info(self.args)

    def _create_model(self):
        '''
        Initialize KG embedding model and optimizer.
        return: model, optimizer
        '''
        model = Controller(self.args, self.kg)
        model.to(self.args.device)
        init_param(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.l2)
        return model, optimizer

    def train(self):
        '''
        Training process
        :return: training time
        '''
        start_time = time.time()
        self.best_valid = 0.0
        self.stop_epoch = 0
        trainer = Trainer(self.args, self.kg, self.model, self.optimizer)


        print("Start Training ===============================>")
        '''Training iteration'''
        for epoch in range(self.start_epoch, int(self.args.epoch_num)):

            
            self.args.epoch = epoch
            '''training'''
            loss, valid_res, num_known_pairs, num_pseudo_pairs, triple_stats = trainer.run_epoch()
            '''early stop'''
            if self.best_valid <= valid_res[self.args.valid_metrics]:
                self.best_valid = valid_res[self.args.valid_metrics]
                self.stop_epoch = max(0, self.stop_epoch-self.args.patience)
                self.save_model(is_best=True)
            else:
                self.stop_epoch += 1
                if self.stop_epoch >= self.args.patience:
                    self.args.logger.info('Early Stopping! Epoch: {} Best Results: {}'.format(epoch, round(self.best_valid*100, 3)))
                    break
            '''logging'''
            if epoch % 1 == 0:
                self.args.logger.info('Epoch:{}\tLoss:{}\tH@1:{}\tH@3:{}\tH@5:{}\tH@10:{}\tMRR:{}\tBest:{}'.format(epoch,round(loss, 3), round(valid_res['hits1'] * 100, 2), round(valid_res['hits3'] * 100, 2), round(valid_res['hits5'] * 100, 2), round(valid_res['hits10'] * 100, 2), round(valid_res['mrr'] * 100, 2), round(self.best_valid * 100,2)))
        end_time = time.time()
        training_time = end_time - start_time
        return training_time

    def test(self, load_best=True):
        self.kg.load_test()
        if load_best and self.args.load_checkpoint is None:
            best_checkpoint = os.path.join(self.args.save_path, 'model_best.tar')
            self.load_checkpoint(best_checkpoint)
        tester = Tester(self.args, self.kg, self.model)

        res = tester.test()
        print(res)
        return res

    def prepare(self):
        '''
        set the log path, the model saving path and device
        :return: None
        '''
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)


        self.args.data_path = args.data_path + args.dataset + '/'
        self.args.save_path = args.save_path + args.dataset + '-' + args.scorer + '-' + args.encoder +'-'+ str(args.emb_dim)+'-' + str(args.margin)

        if self.args.use_known_alignment:
            self.args.save_path = self.args.save_path + '-known_align' + str(self.args.known_align_weight)
        if self.args.use_pseudo_alignment:
            self.args.save_path = self.args.save_path + '-pseudo_align' + str(self.args.pseudo_align_weight)
            if getattr(self.args, 'use_pseudo_align_threshold', True):
                self.args.save_path = self.args.save_path + '-pseudo_align_threshold' + str(self.args.pseudo_align_threshold)
            else:
                self.args.save_path = self.args.save_path + '-no_threshold'
            self.args.save_path = self.args.save_path + '-max_rounds' + str(self.args.pseudo_align_max_rounds)
        if self.args.use_known_expand_triple_loss:
            self.args.save_path = self.args.save_path + '-known_expand_weight' + str(self.args.known_expand_triple_weight)
        if self.args.use_pseudo_triple_loss:
            self.args.save_path = self.args.save_path + '-pseudo_triple_weight' + str(self.args.pseudo_triple_weight)

        if getattr(self.args, 'use_entropy_selection', False):
            self.args.save_path = self.args.save_path + '-entropy' + str(self.args.entropy_threshold) 
        if getattr(self.args, 'use_triple_infonce', False):
            self.args.save_path = self.args.save_path + '-infonce' + str(self.args.triple_infonce_weight) + '-temp' + str(self.args.triple_infonce_temperature)

            if getattr(self.args, 'use_pseudo_triple_infonce', False):
                self.args.save_path = self.args.save_path + '-pseudo_infonce' + str(getattr(self.args, 'pseudo_triple_infonce_weight', 0.1))
    
        self.args.save_path = self.args.save_path + '-' + '--' + str(self.args.learning_rate)+ '-' + str(args.seed) + '-neg_ratio-' + str(args.neg_ratio)
        
        if self.args.note != '':
            self.args.save_path = self.args.save_path + self.args.note

        if os.path.exists(args.save_path) and args.load_checkpoint is None:
            shutil.rmtree(args.save_path, True)
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        self.args.log_path = args.log_path + datetime.now().strftime('%Y%m%d/')
        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)
        self.args.log_path = args.log_path + args.dataset + '-' + args.scorer + '-' + args.encoder +'-'+ str(args.emb_dim)+ '-' + str(args.margin)
        


        if self.args.use_known_alignment:
            self.args.log_path = self.args.log_path + '-known_align' + str(self.args.known_align_weight)
        if self.args.use_pseudo_alignment:
            self.args.log_path = self.args.log_path + '-pseudo_align' + str(self.args.pseudo_align_weight) 
            if getattr(self.args, 'use_pseudo_align_threshold', True):
                self.args.log_path = self.args.log_path + '-pseudo_align_threshold' + str(self.args.pseudo_align_threshold)
            else:
                self.args.log_path = self.args.log_path + '-no_threshold'
            self.args.log_path = self.args.log_path + '-max_rounds' + str(self.args.pseudo_align_max_rounds)
        if self.args.use_known_expand_triple_loss:
            self.args.log_path = self.args.log_path + '-known_expand_weight' + str(self.args.known_expand_triple_weight)
        if self.args.use_pseudo_triple_loss:
            self.args.log_path = self.args.log_path + '-pseudo_triple_weight' + str(self.args.pseudo_triple_weight)
        

        if getattr(self.args, 'use_entropy_selection', False):
            self.args.log_path = self.args.log_path  + '-entropy' + str(self.args.entropy_threshold) 

        
        if getattr(self.args, 'use_triple_infonce', False):
            self.args.log_path = self.args.log_path + '-infonce' + str(self.args.triple_infonce_weight) + '-temp' + str(self.args.triple_infonce_temperature)

            if getattr(self.args, 'use_pseudo_triple_infonce', False):
                self.args.log_path = self.args.log_path + '-pseudo_infonce' + str(getattr(self.args, 'pseudo_triple_infonce_weight', 0.1))


        self.args.log_path = self.args.log_path + '-' + '--' +str(self.args.learning_rate) + '-' + str(args.seed) + '-neg_ratio-' + str(args.neg_ratio)
        
        
        '''add additional note to log name'''
        if self.args.note != '':
            self.args.log_path = self.args.log_path + self.args.note

        '''set logger'''
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        console_formatter = logging.Formatter('%(asctime)-8s: %(message)s')
        logging_file_name = args.log_path + '.txt'
        file_handler = logging.FileHandler(logging_file_name)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = console_formatter
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        self.args.logger = logger

        '''set device'''
        torch.cuda.set_device(int(args.gpu))
        _ = torch.tensor([1]).cuda()
        self.args.device = _.device

    def save_model(self, is_best=False, name=''):
        '''
        Save trained model.
        :param is_best: If True, save it as the best model.
        After training on each snapshot, we will use the best model to evaluate.
        '''
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = self.model.state_dict()
        checkpoint_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        checkpoint_dict['epoch_id'] = self.args.epoch

        if is_best:
            self.args.logger.info('Saving Best Model to {}/model_best.tar'.format(self.args.save_path))
            out_tar = os.path.join(self.args.save_path, 'model_best.tar')
            torch.save(checkpoint_dict, out_tar)

        if name != '':
            out_tar = os.path.join(name)
            torch.save(checkpoint_dict, out_tar)
    def load_checkpoint(self, input_file):
        if os.path.isfile(os.path.join(os.getcwd(), input_file)):
            logging.info('=> loading checkpoint \'{}\''.format(os.path.join(os.getcwd(), input_file)))
            checkpoint = torch.load(os.path.join(os.getcwd(), input_file), map_location="cuda:{}".format(self.args.gpu))
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return int(checkpoint['epoch_id']) + 1
        else:
            logging.info('=> no checkpoint found at \'{}\''.format(input_file))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # training control
    parser.add_argument('-dataset', dest='dataset', default='WIKI-YAGO', help='dataset name, DBP-FB, WIKI-YAGO, ES-EN, JA-EN')
    parser.add_argument('-load_checkpoint', dest='load_checkpoint', default=None, help='./model_best.tar')

    # base setting
    parser.add_argument('-optimizer_name', dest='optimizer_name', default='Adam')
    parser.add_argument('-epoch_num', dest='epoch_num', default=500, type=int, help='max epoch num')
    parser.add_argument('-batch_size', dest='batch_size', default=2048, type=int, help='Mini-batch size')
    parser.add_argument('-test_batch_size', dest='test_batch_size', default=100,type=int, help='Mini-batch size')
    parser.add_argument('-learning_rate', dest='learning_rate', default=0.0005, type=float)
    parser.add_argument('-emb_dim', dest='emb_dim', default=256,type=int,help='embedding dimension')
    parser.add_argument('-l2', dest='l2', default=0.0,type=float, help='optimizer l2')

    parser.add_argument('-patience', dest='patience', default=5, type=int, help='early stop step')
    parser.add_argument('-neg_ratio', dest='neg_ratio', default=256,type=int)
    parser.add_argument('-gpu', dest='gpu', default=0,type=int)
    parser.add_argument('-margin', dest='margin', default=None, type=float, help='')


    parser.add_argument('-encoder', dest='encoder', default='lookup_gat', help='lookup_gat')
    parser.add_argument('-scorer', dest='scorer', default='TransE', help='')
    parser.add_argument('-ea_expand_training', dest='ea_expand_training', default='True', help='')

    
    parser.add_argument('-use_known_alignment', dest='use_known_alignment', default='True', type=str,  help='')
    parser.add_argument('-use_pseudo_alignment', dest='use_pseudo_alignment', default='True', type=str, help='')
    parser.add_argument('-known_align_weight', dest='known_align_weight', default=0.01, type=float, help='')
    parser.add_argument('-pseudo_align_weight', dest='pseudo_align_weight', default=0.01, type=float, help='')
    parser.add_argument('-pseudo_align_threshold', dest='pseudo_align_threshold', default=None, type=float, help='')
    parser.add_argument('-pseudo_align_max_rounds', dest='pseudo_align_max_rounds', default=None, type=int, help='')
    parser.add_argument('-use_pseudo_align_threshold', dest='use_pseudo_align_threshold', default='True', type=str, help='')

    parser.add_argument('-use_known_expand_triple_loss', dest='use_known_expand_triple_loss', default='True', type=str, help='')
    parser.add_argument('-known_expand_triple_weight', dest='known_expand_triple_weight', default=0.1, type=float, help='')
    parser.add_argument('-use_pseudo_triple_loss', dest='use_pseudo_triple_loss', default='True', type=str, help='')
    parser.add_argument('-pseudo_triple_weight', dest='pseudo_triple_weight', default=0.1, type=float, help='')
   
    parser.add_argument('-use_triple_infonce', dest='use_triple_infonce', default='True', type=str, help='')
    parser.add_argument('-triple_infonce_temperature', dest='triple_infonce_temperature', default=None, type=float, help='')
    parser.add_argument('-triple_infonce_weight', dest='triple_infonce_weight', default=0.1, type=float, help='')
    parser.add_argument('-use_pseudo_triple_infonce', dest='use_pseudo_triple_infonce', default='True', type=str, help='')
    parser.add_argument('-pseudo_triple_infonce_weight', dest='pseudo_triple_infonce_weight', default=0.1, type=float, help='')
    
    parser.add_argument('-pseudo_triple_selection', dest='pseudo_triple_selection', default='score_high', type=str, choices=['score_high', 'score_low'], help='')
    
    parser.add_argument('-use_entropy_selection', dest='use_entropy_selection', default='True', type=str, help='')
    parser.add_argument('-entropy_n_sigma', dest='entropy_n_sigma', default=1.0, type=float, help='')
    parser.add_argument('-entropy_threshold', dest='entropy_threshold', default=10.0, type=float, help='')
    parser.add_argument('-use_inter_triples', dest='use_inter_triples', default='True', type=str, help='')

    parser.add_argument('-align_weight', dest='align_weight', default=None, type=float, help='')
    parser.add_argument('-triple_weight', dest='triple_weight', default=None, type=float, help='')
    parser.add_argument('-infonce_weight', dest='infonce_weight', default=None, type=float, help='')

    # others
    parser.add_argument('-save_path', dest='save_path', default='./checkpoint/')
    parser.add_argument('-data_path', dest='data_path', default='CrossLPData/')
    parser.add_argument('-root_dir', dest='root_dir', default='CrossLPData/')
    parser.add_argument('-log_path', dest='log_path', default='./logs/')
    parser.add_argument('-num_workers', dest='num_workers', default=10,type=int)
    parser.add_argument('-seed', dest='seed', default=2024,type=int)
    parser.add_argument('-valid_metrics', dest='valid_metrics', default='mrr')
    parser.add_argument('-note', dest='note', default='develop', help='The note of log file name')
    args = parser.parse_args()

    dataset_best_params = BEST_PARAMS_BY_DATASET.get(args.dataset)
    if dataset_best_params is not None:
        for param_name, param_value in dataset_best_params.items():
            if getattr(args, param_name) is None:
                setattr(args, param_name, param_value)

    if args.align_weight is not None:
        args.known_align_weight = args.align_weight
        args.pseudo_align_weight = args.align_weight
    
    if args.triple_weight is not None:
        args.known_expand_triple_weight = args.triple_weight
        args.pseudo_triple_weight = args.triple_weight
    
    if args.infonce_weight is not None:
        args.triple_infonce_weight = args.infonce_weight
        args.pseudo_triple_infonce_weight = args.infonce_weight

    retype_parameters(args)
    same_seeds(args.seed)

    args.source_list = args.dataset.split('-')
    E = experiment(args)
    E.train()
    E.test()
    


