import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime


class Rossmann(object):
    def __init__(self):
        self.home_path = r'C:\Users\06564176686\repos\Store-Sales-Prediction'
        self.encoding_competition_str        = pickle.load(open(self.home_path + 'parameter\encoding_competition_str.pkl', 'rb'))
        self.encoding_competition_time_month = pickle.load(open(self.home_path + 'parameter\encoding_competition_time_month.pkl', 'rb'))
        self.encoding_promo_time_week        = pickle.load(open(self.home_path + 'parameter\encoding_promo_time_week.pkl', 'rb'))
        self.encoding_year                   = pickle.load(open(self.home_path + 'parameter\encoding_year.pkl', 'rb'))
        self.encoding_store_type             = pickle.load(open(self.home_path + 'parameter\encoding_store_type.pkl', 'rb'))


    def data_cleaning(self, df1):
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
       'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
       'CompetitionDistance', 'CompetitionOpenSinceMonth',
       'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
       'Promo2SinceYear', 'PromoInterval']

        snakecase = lambda x: inflection.underscore(x)
        cols_new = list(map(snakecase, cols_old))
        
        # rename
        df1.columns = cols_new
        
        df1['date'] = pd.to_datetime(df1['date'])
        
        #competition_distance              2642
        # Imaginar que não há competidores próximos, ou seja, distância grande.
        faraway = 100*df1.competition_distance.max()
        near = 50

        df1['competition_distance'].fillna(faraway, inplace=True)
        df1['competition_distance'] = df1['competition_distance'].apply(lambda x: near if x <=near else x)

        # A ideia aqui é aplicar o data atual (do dado lido) para dizer depois que não há tempo com concorrente.
        # applicar df1.date para os campos de competition open since

        #competition_open_since_month    323348
        df1['competition_open_since_month']= df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis=1)

        #competition_open_since_year     323348
        df1['competition_open_since_year']= df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis=1)
        
        
        # Atribuir a data atual (da leitura) para dizer que não há tempo desde a última promo.
        #promo2_since_week              508031

        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)

        #promo2_since_year               508031
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)


        #promo_interval  

        # criar month map dict
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

        # criar colunas 'month_map' com base em date
        df1['month_map'] = df1['date'].dt.month.map(month_map)

        # criar 'is_promo' comparando o map com promo_inverval
        df1['promo_interval'].fillna('', inplace=True)


        df1['is_promo'] = df1[['promo_interval', 'month_map']].apply(lambda x: 1 if x['month_map'] in x['promo_interval'].split(',')
                                    else 0 ,axis=1)

        
        # competiton
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype( int64 )
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype( int64 )

        # promo2
        df1['promo2_since_week'] = df1['promo2_since_week'].astype( int64 )
        df1['promo2_since_year'] = df1['promo2_since_year'].astype( int64 )
        
        return df1
    
    def feature_engineering( self, df2):

        # Derivando datas como variáveis categóricas
        # year
        df2['year'] = df2['date'].dt.year

        # month
        df2['month'] = df2['date'].dt.month

        # day
        df2['day'] = df2['date'].dt.day

        # week of year
        df2['week_of_year'] = df2['date'].dt.weekofyear

        # year week
        df2['year_week'] = df2['date'].dt.strftime( '%Y-%W' )

        # competition since

        # criando uma única feature para dizer a data de início da promoção
        df2['competition_since'] = df2.apply( lambda x: datetime.datetime( year=x['competition_open_since_year'], month=x['competition_open_since_month'],day=1 ), axis=1 )

        # calcula o timedelta do início da promoção para a data do registro
        df2['competition_time_month'] = ( ( df2['date'] - df2['competition_since'] )/30 ).apply( lambda x: x.days ).astype( int64 )

        # promo since

        # cria uma str com o formato da data
        df2['promo_since'] = df2['promo2_since_year'].astype( str ) + '-' + df2['promo2_since_week'].astype( str )

        # transforma a str em data
        df2['promo_since'] = df2['promo_since'].apply( lambda x: datetime.datetime.strptime( x + '-1', '%Y-%W-%w' ) - datetime.timedelta( days=7 ) )

        # calcula o timedelta da promoção em semanas
        df2['promo_time_week'] = ( ( df2['date'] - df2['promo_since'] )/7 ).apply( lambda x: x.days ).astype( int64 )



        # competition strenght - indica de forma mais direta a interferência da competição

        df2['competition_str'] = 1/df2['competition_distance']


        ## PASSO 03 - FILTRAGEM DE VARIÁVEIS


        # Exclui linhas quando as lojas estavam fechadas ou não houve vendas
        df2 = df2[(df2['open'] != 0)]

        ## Feature Selection

        #Excluindo features que não fazem sentido para modelagem
        cols_drop = ['open', 'promo_interval', 'month_map', 'competition_open_since_year',
                     'competition_open_since_month', 'competition_since', 'promo_since', 'promo2_since_year',
                     'promo2_since_week', 'competition_distance' ]
        df2 = df2.drop( cols_drop, axis=1 )

        return df2
    
    def data_preparation(self, df5):
    
        ## Rescaling

        # competition_str
        df5['competition_str'] = self.encoding_competition_str.fit_transform(df5[['competition_str']].values)

        # competition time_month
        df5['competition_time_month'] = self.encoding_competition_time_month.fit_transform(df5[['competition_time_month']])

        # promo_time_week
        df5['promo_time_week'] = self.encoding_promo_time_week.fit_transform(df5[['promo_time_week']])

        # year
        df5['year'] = self.encoding_year.fit_transform(df5[['year']])


        ## Encoding

        # state_holiday - One Hot Encoding
        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])

        # store_type - Label Encoding
        le = LabelEncoder()
        df5['store_type'] = self.encoding_store_type.fit_transform(df5['store_type'])

        # assortment - Ordinal Encoding
        assortment_dict = {'a':1,'b':2,'c':3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)

        ## Transformations

        ### Nature transformation for cyclical variables

        # day of week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x : np.sin(x*(2.*np.pi/7)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x : np.cos(x*(2.*np.pi/7)))

        # month
        df5['month_sin'] = df5['month'].apply(lambda x : np.sin(x*(2.*np.pi/12)))
        df5['month_cos'] = df5['month'].apply(lambda x : np.cos(x*(2.*np.pi/12)))

        # day
        df5['day_sin'] = df5['day'].apply(lambda x : np.sin(x*(2.*np.pi/30)))
        df5['day_cos'] = df5['day'].apply(lambda x : np.cos(x*(2.*np.pi/30)))

        # week of year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x : np.sin(x*(2.*np.pi/52)))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x : np.cos(x*(2.*np.pi/52)))
        
        cols_selected_boruta = ['store',
                            'promo',
                            'store_type',
                            'assortment',
                            'promo2',
                            'competition_time_month',
                            'promo_time_week',
                            'competition_str',
                            'day_of_week_sin',
                            'day_of_week_cos',
                            'month_cos',
                            'day_sin',
                            'day_cos',
                            'week_of_year_cos'
                           ]
        
        return df5[cols_selected_boruta]

    def get_prediction(self, model, original_data, test_data):
        #prediction
        pred = model.predict(test_data)
        
        # join pred with original data
        original_data['prediction'] = np.empm1(pred)
                  
        return original_data.to_json(orient='records', date_format = 'iso')