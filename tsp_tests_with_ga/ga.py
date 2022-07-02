import numpy as np
import pandas as pd
#from data_generation import distance_generation
import matplotlib.pyplot as plt
def repeat(a):
    a=list(a)
    for i in a:
        if a.count(i)>1:
            return True
    return False
'''
def repeat_executation(a,b):
    pass##############
'''


class Ga:
    def __init__(self,size,iteration,city_num,crossover_rate,mutation_rate,crossover_rate_min,crossover_rate_max):
        self.iteration=iteration
        self.size=size
        self.city_num=city_num
        self.population=np.random.randint(5,size=(self.size,self.city_num))#zeros使得数据类型都是小数 不方便
        self.population.astype(np.int32)
        self.distance_matrix=np.zeros((self.city_num,self.city_num))
        self.fitness=np.zeros(self.size)
        self.crossover_rate=crossover_rate
        self.crossover_rate_min=crossover_rate_min
        self.crossover_rate_max=crossover_rate_max
        self.mutation_rate=mutation_rate


        #vint=np.vectorize(int)
        for i in range(self.size):
            self.population[i,:]=np.random.permutation(self.city_num)
            #self.population[i,:]=vint(self.population[i,:])
            self.population[i,:].astype(np.int32)
        self.best_individual=self.population[0]






    def fitness_calculation(self):
        for i in range(self.size):
            x_temp=self.population[i]
            sum_temp=0
            for j in range(self.city_num-1):
                start=int(x_temp[j])
                end=int(x_temp[j+1])
                sum_temp+=self.distance_matrix[start,end]
            sum_temp+=self.distance_matrix[x_temp[-1]][x_temp[0]]#最后加上从末尾到首端的距离
            self.fitness[i]=sum_temp

    def temp_fitness_calculation(self,temp_individual):
        x_temp = temp_individual
        sum_temp = 0
        for j in range(self.city_num - 1):
            start = int(x_temp[j])
            end = int(x_temp[j + 1])
            sum_temp += self.distance_matrix[start, end]
        sum_temp += self.distance_matrix[x_temp[-1]][x_temp[0]]  # 最后加上从末尾到首端的距离
        return sum_temp



    def choice(self):
        self.fitness_calculation()#原先问题之一在于忘记在选择操作前先进行适应度计算更新 总是用第一次的适应度
        fitness_temp=1/self.fitness
        fitness_temp=fitness_temp/fitness_temp.sum()
        choice_temp=np.random.choice(a=np.arange(self.size),size=self.size,p=fitness_temp)
        #print(choice_temp)
        self.population=self.population[choice_temp]
        self.fitness_calculation()#选择完成后种群发生变化 适应度重新计算
        best_fitness_temp = self.fitness.min()#选择操作完成后选出最优个体 便于后面精英操作
        best_idx_temp=np.where(self.fitness==best_fitness_temp)[0][0]
        self.best_individual=self.population[best_idx_temp].copy()

    def elite(self):#精英策略 在交叉和变异后 需要重新计算适应度
        self.fitness_calculation()
        worst_fitness_temp=self.fitness.max()
        worst_idx_temp = np.where(self.fitness == worst_fitness_temp)[0][0]
        self.population[worst_idx_temp]=self.best_individual



    def crossover(self):
        for I in range(int(self.size/2)):
            if np.random.rand()>self.crossover_rate:
                continue
            temp1=self.population[2*I].copy()
            temp2=self.population[2*I+1].copy()
            temp3=np.random.randint(self.city_num)
            temp4=np.random.randint(self.city_num)
            crosspoint1=min(temp3,temp4)
            crosspoint2=max(temp3,temp4)
            #print('crosspoint1',crosspoint1)
            #print('crosspoint2',crosspoint2)
            temp_dict1={};temp_dict2={}
            for j in range(crosspoint1,crosspoint2):
                temp_dict1[int(temp1[j])]=int(temp2[j])#不知为何字典存储与设想的相反
                temp_dict2[int(temp2[j])]=int(temp1[j])
            temp1[crosspoint1:crosspoint2],temp2[crosspoint1:crosspoint2]=temp2[crosspoint1:crosspoint2].copy(),temp1[crosspoint1:crosspoint2].copy()#深浅拷贝问题
            #temp5=temp1[crosspoint1:crosspoint2]
            #temp1[crosspoint1:crosspoint2]=temp2[crosspoint1:crosspoint2]
            #temp2[crosspoint1:crosspoint2]=temp5
            #print(temp1)
            #print(temp2)

            while repeat(temp1):
                for i in temp1[crosspoint1:crosspoint2]:
                    i=int(i)
                    if list(temp1).count(i)>1:
                        idx_temp=np.where(temp1==i)[0]
                        for t in idx_temp:
                            if t not in range(crosspoint1,crosspoint2):
                                idx=t
                        #print('run here')
                        #print(temp1)
                        #print(temp2)
                        #print(idx)
                        #print(temp_dict1)
                        #print(temp_dict2)
                        #print(i)
                        temp1[idx]=temp_dict2[i]
            while repeat(temp2):
                for i in temp2[crosspoint1:crosspoint2]:
                    if list(temp2).count(i)>1:
                        idx_temp=np.where(temp2==i)[0]
                        for t in idx_temp:
                            if t not in range(crosspoint1,crosspoint2):
                                idx=t
                        temp2[idx]=temp_dict1[i]
            self.population[2*I]=temp1
            self.population[2*I+1]=temp2


    def self_adapation_crossover(self):#交叉概率与当前适应度相关 其余部分与普通的交叉相同 当前问题适应度越低越好 因此适应度越低交叉概率越低 避免造成破坏
        for I in range(int(self.size / 2)):
            fitness_mean_temp=self.fitness.mean()
            temp1 = self.population[2 * I].copy()
            temp2 = self.population[2 * I + 1].copy()
            temp1_fitness=self.temp_fitness_calculation(temp_individual=temp1)
            temp2_fitness = self.temp_fitness_calculation(temp_individual=temp2)
            temp_fitness=min(temp1_fitness,temp2_fitness)
            if temp_fitness>fitness_mean_temp:
                crossover_rate_temp=self.crossover_rate_max
            else:
                crossover_rate_temp=self.crossover_rate_min+(self.crossover_rate_max-self.crossover_rate_min)*(temp_fitness-self.fitness.min())/(fitness_mean_temp-self.fitness.min())#########

            #if np.random.rand() > self.crossover_rate:
            if np.random.random()>crossover_rate_temp:
                continue
            #temp1 = self.population[2 * I].copy()
            #temp2 = self.population[2 * I + 1].copy()
            temp3 = np.random.randint(self.city_num)
            temp4 = np.random.randint(self.city_num)
            crosspoint1 = min(temp3, temp4)
            crosspoint2 = max(temp3, temp4)
            # print('crosspoint1',crosspoint1)
            # print('crosspoint2',crosspoint2)
            temp_dict1 = {};
            temp_dict2 = {}
            for j in range(crosspoint1, crosspoint2):
                temp_dict1[int(temp1[j])] = int(temp2[j])  # 不知为何字典存储与设想的相反
                temp_dict2[int(temp2[j])] = int(temp1[j])
            temp1[crosspoint1:crosspoint2], temp2[crosspoint1:crosspoint2] = temp2[
                                                                             crosspoint1:crosspoint2].copy(), temp1[
                                                                                                              crosspoint1:crosspoint2].copy()  # 深浅拷贝问题
            # temp5=temp1[crosspoint1:crosspoint2]
            # temp1[crosspoint1:crosspoint2]=temp2[crosspoint1:crosspoint2]
            # temp2[crosspoint1:crosspoint2]=temp5
            # print(temp1)
            # print(temp2)

            while repeat(temp1):
                for i in temp1[crosspoint1:crosspoint2]:
                    i = int(i)
                    if list(temp1).count(i) > 1:
                        idx_temp = np.where(temp1 == i)[0]
                        for t in idx_temp:
                            if t not in range(crosspoint1, crosspoint2):
                                idx = t
                        # print('run here')
                        # print(temp1)
                        # print(temp2)
                        # print(idx)
                        # print(temp_dict1)
                        # print(temp_dict2)
                        # print(i)
                        temp1[idx] = temp_dict2[i]
            while repeat(temp2):
                for i in temp2[crosspoint1:crosspoint2]:
                    if list(temp2).count(i) > 1:
                        idx_temp = np.where(temp2 == i)[0]
                        for t in idx_temp:
                            if t not in range(crosspoint1, crosspoint2):
                                idx = t
                        temp2[idx] = temp_dict1[i]
            self.population[2 * I] = temp1
            self.population[2 * I + 1] = temp2


    def mutation(self):
        for i in range(self.size):
            if np.random.rand()>self.mutation_rate:
                continue
            #print('变异个体',i)
            temp=self.population[i].copy()
            temp1=np.random.randint(self.city_num)
            temp2=np.random.randint(self.city_num)
            mutationpoint1=min(temp1,temp2)
            mutationpoint2=max(temp1,temp2)
            #print('交换次序点1',mutationpoint1)
            #print('交换次序点2',mutationpoint2)
            temp[mutationpoint1],temp[mutationpoint2]=temp[mutationpoint2],temp[mutationpoint1]
            self.population[i]=temp##之前的问题在于忘记将设置的变异相关操作实际执行















test=Ga(size=20,iteration=5000,city_num=30,crossover_rate=0.8,crossover_rate_min=0.2,crossover_rate_max=1,mutation_rate=0.2)
#print(test.population)
#distance_matrix,x,y=distance_generation()
data=pd.read_excel('data_generation.xlsx')
x=data['x']
y=data['y']
distance_matrix=np.zeros((len(x),len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        distance_matrix[i, j] = ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) ** 0.5
test.distance_matrix=distance_matrix
#test.crossover_rate=0.6
#test.mutation_rate=0.2
#test.size=100
'''
test.fitness_calculation()

test.choice()
test.fitness_calculation()
#print(test.population[:2])
test.crossover()
test.mutation()
#print('after crossover')
#print(test.population[:2])
'''
test.fitness_calculation()
temp=test.fitness.min()
best_individual_list=[]
best_individual_idx=np.where(test.fitness==test.fitness.min())[0][0]
best_individual=test.population[best_individual_idx]

for i in range(test.iteration):
    test.fitness_calculation()
    #temp=test.fitness.min()
    if temp>test.fitness.min():#不如上一代最佳的个体就使用之前的最佳
        temp=test.fitness.min()
        best_individual_idx=np.where(test.fitness==test.fitness.min())[0][0]
        best_individual=test.population[best_individual_idx]
    best_individual_list.append(temp)
    #test.fitness_calculation()
    test.choice()
    #test.crossover()
    test.self_adapation_crossover()#自适应交叉
    test.mutation()
    test.elite()#精英策略
plt.figure()
plt.plot(best_individual_list)
plt.show()
print('第一代',best_individual_list[0])
print('历史最佳',np.min(best_individual_list))
#print('最后一代',best_individual_list[-1])
print('历史最佳个体',best_individual)
plt.figure()
for i in range(len(best_individual)-1):
    idx1=best_individual[i]
    idx2=best_individual[i+1]
    plt.plot([x[idx1],x[idx2]],[y[idx1],y[idx2]])
    plt.scatter([x[idx1],x[idx2]],[y[idx1],y[idx2]])
plt.plot([x[best_individual[-1]],x[best_individual[0]]],[y[best_individual[-1]],y[best_individual[0]]])#将末端与首端连接
plt.scatter([x[best_individual[-1]],x[best_individual[0]]],[y[best_individual[-1]],y[best_individual[0]]])
plt.show()









