import torch
import time
b = 5
n = 10
# torch.manual_seed(1234)


ts = torch.abs(torch.randn(b,5,169,64))
tt = torch.abs(torch.randn(b,5,169,64))
seen_vec = torch.abs(torch.randn(n,64))
# ts = torch.Tensor([[1,1,1],[1,0,0]])
# seen_vec = torch.Tensor([[1,1,0],[1,1,1]])

def cos_sim(vector1,vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a,b in zip(vector1,vector2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB==0.0:
        return 0
    else:
        return dot_product / ((normA*normB)**0.5)


# t1 = time.clock()
# ts = ts.view(-1,64)
# noobj_sim = torch.zeros(b*5*169,1)
# # noobj_sim = torch.zeros(2)
# for i in range(seen_vec.shape[0]):
#     for j in range(ts.shape[0]):
#         sim = cos_sim(ts[j],seen_vec[i])
#         # print(sim)
#         if sim > noobj_sim[j]:
#             noobj_sim[j] = sim
# print(noobj_sim)
# t2 = time.clock()
# tf = t2-t1
# print(tf)

# print(tf)

# t1 = time.clock()
# #########3 method 2 ###############
# tsb = ts.view(b, -1 ,64,1).repeat(1,1,1,n).view(-1,64) # [b,845,64,2]
# tsb = ts.view(-1,3,1).repeat(1,1,2).view(-1,3)
# # seen_attrs = torch.zeros( b, 5 * 169, 64, n, requires_grad=False)
# seen_attrs = torch.zeros(2,3,2)
# for i,seen_attr in enumerate(torch.FloatTensor(seen_vec)):
#     # seen_attrs[:,:,:,i] = seen_attr.view(1,1,64).repeat(b,5*169,1)
#     seen_attrs[:,:,i] = seen_attr.view(1,3).repeat(2,1)
# seen_attrs = seen_attrs.view(-1, 3)
# print(seen_attrs)
#
# noobj_sim = torch.zeros(tsb.shape[0], requires_grad=False)
# for i in range(tsb.shape[0]):
#     noobj_sim[i] = cos_sim(tsb[i], seen_attrs[i])
# noobj_sim = noobj_sim.view(-1,n)
# print(noobj_sim)
# noobj_sim, _ = noobj_sim.max(1)
# t2 = time.clock()
# noobj_sim = noobj_sim.view(b,5,169)
# print(noobj_sim)
# tsen = t2-t1
# print(tsen)
#
# print(tsen-tf)

# ts = torch.Tensor([[1,1,1],[1,0,0]])
# seen_vec = torch.Tensor([[1,1,0],[1,1,1],[1,0,1]])
# ts = ts.view(2,3,1).repeat(1,1,3)
# seen = torch.zeros(2,3,3)
# print(seen.shape)

t1=time.clock()
tsb = ts.view(b, -1 ,64,1).repeat(1,1,1,n) # [b,845,64,2]
seen_attrs = torch.zeros( b, 5 *169, 64, n, requires_grad=False)
for i,seen_attr in enumerate(torch.FloatTensor(seen_vec)):
    seen_attrs[:,:,:,i] = seen_attr.view(1,1,64).repeat(b,5*169,1)

print(tsb.shape,seen_attrs.shape)

cos_sim = torch.nn.CosineSimilarity(dim=2,eps=1e-8)
loss = cos_sim(tsb,seen_attrs)
noobj_sim, _ = loss.max(2)
print(noobj_sim.shape)
t2=time.clock()
tbb = t2-t1
print(tbb)



