#include<bits/stdc++.h>
#include<math.h>
using namespace std;
ifstream fin;
ofstream fout;
//float gamma;
struct neuron
{
    int layerno;
    float input;
    vector<float> in_weights;
    float output;
    float error;
    vector<float> delta_weight;
};

//Initialise the weights of the network randomly
void initialise_network(int nh,int nl,int features,vector<vector<neuron>> &network)
{
    vector<neuron> t;
    for(int i=0;i<nl;i++)
    {
        neuron n;
        for(int j=0;j<features;j++)
        {
            float r = (float) rand()/RAND_MAX;
            n.in_weights.push_back(r);
        }
        n.layerno=0;
        t.push_back(n);
    }
    network.push_back(t);
    for(int k=1;k<nh;k++)
    {
        vector<neuron> temp;
        for(int i=0;i<nl;i++)
        {
            neuron n;
            for(int j=0;j<nl;j++)
            {
                float r = (float) rand()/RAND_MAX;
                n.in_weights.push_back(r);
            }
            n.layerno=k;
            temp.push_back(n);
        }
        network.push_back(temp);
    }
    neuron n;
    vector<neuron> temp;
    for(int i=0;i<nl;i++)
    {
        float r = (float) rand()/RAND_MAX;
        n.in_weights.push_back(r);
    }
    n.layerno=nh;
    temp.push_back(n);
    network.push_back(temp);
    
}

//if the network chosen by user is not fully connected then this function deletes the connections that are not given input by user
void type_of_network(vector<vector<neuron>> &network,vector<vector<vector<int>>> connection_matrix,int nl)
{
    for(int i=0;i<connection_matrix.size();i++)
    {
        for(int j=0;j<connection_matrix[i].size();j++)
        {
            int count=0;
            for(int k=0,l=0;k<nl;)
            {
            	if(l<connection_matrix[i][j].size())
		         {
		            if(connection_matrix[i][j][l]==k)
		            {
		                k++;
		                l++;
		            }
		            else
		            {
		            	network[i+1][k].in_weights[j]=0;
		            	k++;
		            }
		         }
		         else
		         {
		         	network[i+1][k].in_weights[j]=0;
		            k++;
		         }
            }
        }
    }
    
}
//Activation functions and their derivatives
float sigmoid_func(float x)
{
    return (1/(1+exp(-x)));
}

float sigmoid_derv(float y)
{
    return y*(1-y);
}

float relu_func(float x)
{
    if(x<=0)
    {
        return 0;
    }
    else
    {
        return x;
    }
}

float relu_derv(float x)
{
    if(x<=0)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

float linear_func(float x)
{
    return x;
}

float linear_derv()
{
    return 1;
}

float tanh_func(float x)
{
    return (2/(1+exp(-2*x)) - 1);
}

float tanh_derv(float y)
{
    return (1-y*y);
}
//Activation and derivative calling functions
int activation_func(string choice,float x,float &op)
{
    if(choice=="Sigmoid")
    {
        op=sigmoid_func(x);
        return 1;
    }
    else if(choice=="Relu")
    {
        op=relu_func(x);
        return 1;
    }
    else if(choice=="Linear")
    {
        op=linear_func(x);
        return 1;
    }
    else if(choice=="tanh")
    {
        op=tanh_func(x);
        return 1;
    }
    else
    {
    	cout<<"Wrong choice entered"<<endl;
    	return 0;
    }
}

int derivative(string choice,float x,float &op,float y)
{
    if(choice=="Sigmoid")
    {
        op=sigmoid_derv(y);
        return 1;
    }
    else if(choice=="Relu")
    {
        op=relu_derv(x);
        return 1;
    }
    else if(choice=="Linear")
    {
        op=linear_derv();
        return 1;
    }
    else if(choice=="tanh")
    {
        op=tanh_derv(y);
        return 1;
    }
    else
    {
    	cout<<"Wrong choice entered"<<endl;
    	return 0;
    }
}


//Combined input to be given to a neuron in activation function
float combined_input(neuron n,vector<float> input)
{
    float sum=0;
    for(int i=0;i<input.size();i++)
    {
        sum=sum+input[i]*n.in_weights[i];
    }
    return sum;
}

//Overfitting function - Weight decay method applied
float overfitting(neuron n,float gamma)
{
    float sum=0;
    for(int i=0;i<n.in_weights.size();i++)
    {
        sum=sum+n.in_weights[i];
        
    }
    return 2*gamma*sum;
}

//Updating weights after backpropagation
void updateWeight(vector<vector<neuron>> &network)
{
    for(int i=0;i<network.size();i++)
    {
        for(int j=0;j<network[i].size();j++)
        {
            for(int k=0;k<network[i][j].in_weights.size();k++)
            {
                if(network[i][j].in_weights[k]!=0)
                {
                    network[i][j].in_weights[k]+=network[i][j].delta_weight[k];
                }
            }
            network[i][j].delta_weight.clear();
        }
    }
}

//Backpropagation function
void backpropagation(vector<vector<neuron>> &network,float learning_rate,float expected_op,int nl,int nh,string choice,float gamma,int features,vector<float> input)
{
    int op_layer=nh;
    float d;
    neuron n=network[op_layer][0];
    if(derivative(choice,n.input,d,n.output))
    {
    	
    	n.error=(expected_op-n.output)*d - overfitting(n,gamma);
    }
    
    for(int i=0;i<nl;i++)
    {
        n.delta_weight.push_back(learning_rate*n.error*network[op_layer-1][i].output);
        
    }
    network[op_layer][0]=n;
    d=0;
    for(int i=nh-1;i>0;i--)
    {
        for(int j=0;j<nl;j++)
        {
            float sum=0;
            for(int k=0;k<network[i+1].size();k++)
            {
                sum=sum+network[i+1][k].in_weights[j]*network[i+1][k].error;
            }
            if(derivative(choice,network[i][j].input,d,network[i][j].output))
            {
            	network[i][j].error=sum*d - overfitting(network[i][j],gamma);
            	
            }
            
            for(int l=0;l<nl;l++)
            {
                network[i][j].delta_weight.push_back(learning_rate*network[i][j].error*network[i-1][l].output);
                
            }
        }
        
    }
    for(int j=0;j<nl;j++)
    {
    	float sum=0;
    	for(int k=0;k<network[1].size();k++)
        {
            sum=sum+network[1][k].in_weights[j]*network[1][k].error;
            
        }
        if(derivative(choice,network[0][j].input,d,network[0][j].output))
        {
        	network[0][j].error=sum*d - overfitting(network[0][j],gamma);
        	
        }
        
        for(int l=0;l<features;l++)
        {
            network[0][j].delta_weight.push_back(learning_rate*network[0][j].error*input[l]);
            
        }
    }
    

}

//4 parameters to be taken input from user
void user_input(vector<vector<neuron>> &network,int &nh,int &nl,string &choice,int features)
{
    int ch;
    cout<<"Enter no. of hidden layers : ";
    cin>>nh;
    cout<<"Enter no. of neurons in each layer : ";
    cin>>nl;
    cout<<"Activation function available :\n1.Sigmoid\n2.Relu\n3.Linear\n4.tanh\nEnter choice : ";
    cin>>choice;
    cout<<"Do you want fully connected network (1) or specify your own network (2) ?\nEnter choice 1 or 2 : ";
    cin>>ch;
    initialise_network(nh,nl,features,network);
    if(ch==2)
    {
        vector<vector<vector<int>>> connection_matrix;
        for(int i=0;i<nh-1;i++)
        {
            cout<<"For layer "<<i<<" :"<<endl;
            vector<vector<int>> layer;
            for(int j=0;j<nl;j++)
            {
                
                int n;
                cout<<"Enter no. of connections for neuron "<<j<<" : ";
                cin>>n;
                cout<<"Enter connections : "<<endl;
                vector<int> temp;
                for(int k=0;k<n;k++)
                {
                    int x;
                    cin>>x;
                    temp.push_back(x);
                }
                layer.push_back(temp);
            }
            connection_matrix.push_back(layer);
        }
        type_of_network(network,connection_matrix,nl);
    }
    for(int i=0;i<network.size();i++)
    {
    	cout<<"Layer "<<i<<" :\n";
    	for(int j=0;j<network[i].size();j++)
    	{
    		cout<<"Neuron "<<j<<" :\n";
    		for(int k=0;k<network[i][j].in_weights.size();k++)
    		{
    			cout<<network[i][j].in_weights[k]<<" ";
    		}
    		cout<<endl;
    	}
    	cout<<endl;
    }
}

//Reading the dataset for training
void readDataset(int &features,vector<vector<float>> &dataset)
{
    features=2;
    int rows=10;
    fin.open("input2.txt");
	for(int i=0;i<rows;i++)
	{
	    vector<float> temp;
	    for(int j=0;j<=features;j++)
	    {
	        float x;
	        fin>>x;
	        temp.push_back(x);
	    }
	    dataset.push_back(temp);
	}
	
	fin.close();
}
//Feedforward function
void feedforward(vector<float> input,vector<vector<neuron>> &network,string choice)
{
    for(int i=0;i<network.size();i++)
    {
        vector<float> in=input;
        input.clear();
        
        for(int j=0;j<network[i].size();j++)
        {
            float comin=combined_input(network[i][j],in);
            network[i][j].input=comin;
            float output;
            if(activation_func(choice,comin,output))
            {
                network[i][j].output=output;
                input.push_back(output);
            }
            else
            {
                cout<<"Wrong choice for activation function entered!!!";
            }
        }
    }
}
//Train model
void train_model(vector<vector<neuron>> &network,int epochs,float learning_rate,vector<vector<float>> dataset,string choice,int nl,int nh,float gamma,int features)
{
	cout<<"Features : "<<features<<endl;
    for(int i=0;i<epochs;i++)
    {
		if(i<epochs/4)
		{
			gamma=0.001;
		}
		else if(i<epochs/2)
		{
			gamma=0.005;
		}
		else 
		{
			gamma=0.01;
		}
        float error=0;
        for(int j=0;j<dataset.size();j++)
        {
            float expected_op=dataset[j][features];
            feedforward(dataset[j],network,choice);
            backpropagation(network,learning_rate,expected_op,nl,nh,choice,gamma,features,dataset[j]);
            
            error=error+abs(expected_op - network[nh][0].output);
            updateWeight(network);
            
            
        }
        error=error/dataset.size();
        cout<<"Epoch "<<i+1<<" , error : "<<error<<endl;
        
    }
    cout<<"New network:"<<endl;
        for(int i=0;i<network.size();i++)
		{
			cout<<"Layer "<<i<<" :\n";
			for(int j=0;j<network[i].size();j++)
			{
				cout<<"Neuron "<<j<<" :\n";
				for(int k=0;k<network[i][j].in_weights.size();k++)
				{
					cout<<network[i][j].in_weights[k]<<" ";
				}
				cout<<endl;
			}
			cout<<endl;
		}
}
//Prediction function
void predict_instance(string choice,vector<vector<neuron>> &network,int features,int nh)
{
    /*cout<<"Enter the new instance : ";
    vector<float> instance;
    for(int i=0;i<features;i++)
    {
        float x;
        cin>>x;
        instance.push_back(x);
    }
    feedforward(instance,network,choice);
    cout<<"Predicted output : "<<round(network[nh][0].output)<<endl;*/
    //vector<vector<float>> test;
    fin.open("input2.txt");
    for(int i=0;i<10;i++)
    {
    	vector<float> temp;
    	for(int j=0;j<features;j++)
    	{
    		float a;
    		fin>>a;
    		temp.push_back(a);
    	}
    	float out;
    	fin>>out;
    	feedforward(temp,network,choice);
    	cout<<"Predicted output : "<<round(network[nh][0].output)<<" Expected : "<<out<<endl;
    }
    fin.close();
}
//Start of the program
void execution()
{
    int nh,nl,features,epochs;
    float learning_rate,gamma;
    string choice;
    vector<vector<neuron>> network;
    vector<vector<float>> dataset;
    learning_rate=0.5;
    gamma=0;
    epochs=150;
    readDataset(features,dataset);
    user_input(network,nh,nl,choice,features);
    train_model(network,epochs,learning_rate,dataset,choice,nl,nh,gamma,features);
    predict_instance(choice,network,features,nh);
}

int main()
{
    execution();
    return 0;
}
