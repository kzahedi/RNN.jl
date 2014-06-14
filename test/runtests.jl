using Base.Test
using RNN

#
# test initialisation 
#
rnn = RNN_t(1,2,3)
@test size(rnn.w) == (6,6)
@test size(rnn.a) == (6,1)
@test size(rnn.b) == (6,1)
@test size(rnn.o) == (6,1)
@test rnn.ni      == 1
@test rnn.no      == 2
@test rnn.nh      == 3
@test rnn.n       == 6

#
# test modify synapse on a net without synapses
#
rnn = RNN_t(1,2,3)
rnn_modify_synapse!(rnn, 1.1, 100.0, 10.0)
for i=1:rnn.n
  for j=1:rnn.n
    @test rnn.w[i,j] == 0.0
  end
end

#
# test add synapses
#
rnn = RNN_t(1,2,3)
rnn_add_synapse!(rnn, 1.1, 1.0)
for i=1:rnn.n
  for j=1:rnn.n
    if i <= rnn.ni
      @test rnn.w[i,j] ==  0.0
    else
      @test rnn.w[i,j] !=  0.0
    end
    @test rnn.w[i,j] <=  1.0
    @test rnn.w[i,j] >= -1.0
  end
end


#
# test remove synapse
#
rnn = RNN_t(1,2,3)
rnn_remove_synapse!(rnn, 1.1)
for i=1:rnn.n
  for j=1:rnn.n
    @test rnn.w[i,j] == 0.0
  end
end

#
# test modify synapse
#
rnn1 = RNN_t(1,2,3)
rnn_add_synapse!(rnn1, 1.1, 1.0)
rnn2 = rnn_copy(rnn1)
rnn_modify_synapse!(rnn1, 1.1, 5.0, 10.0)
for i=rnn.ni+1:rnn.n
  for j=1:rnn.n
    @test rnn1.w[i,j] != rnn2.w[i,j]
  end
end
for i=1:rnn.ni
  for j=1:rnn.n
    @test rnn1.w[i,j] == 0.0
    @test rnn2.w[i,j] == 0.0
  end
end

#
# test add neuron
#
rnn = RNN_t(1,2,3)
rnn_add_neuron!(rnn, 1.0, 1.0, 2.0, 1.0, 4)
rnn_add_neuron!(rnn, 1.0, 1.0, 2.0, 1.0, 4)
rnn_add_neuron!(rnn, 1.0, 1.0, 2.0, 1.0, 4)
@test rnn.n       == 7
@test rnn.ni      == 1
@test rnn.no      == 2
@test rnn.nh      == 4
@test size(rnn.w) == (7,7)
@test size(rnn.a) == (7,1)
@test size(rnn.b) == (7,1)
@test size(rnn.o) == (7,1)
for i=rnn.ni+1:rnn.n
  @test rnn.w[i,rnn.n] != 0.0
end
for i=1:rnn.ni
  @test rnn.w[i,rnn.n] == 0.0
end
for i=1:rnn.n
  @test rnn.w[rnn.n,i] != 0.0
end
rnn_add_neuron!(rnn, 1.0, 1.0, 2.0, 1.0, 8)
rnn_add_neuron!(rnn, 1.0, 1.0, 2.0, 1.0, 8)
rnn_add_neuron!(rnn, 1.0, 1.0, 2.0, 1.0, 8)
rnn_add_neuron!(rnn, 1.0, 1.0, 2.0, 1.0, 8)
@test rnn.n       == 11
@test rnn.ni      == 1
@test rnn.no      == 2
@test rnn.nh      == 8
@test size(rnn.w) == (11,11)
@test size(rnn.a) == (11,1)
@test size(rnn.b) == (11,1)
@test size(rnn.o) == (11,1)

#
# test modify bias
#
rnn = RNN_t(1,2,3)
rnn_modify_bias!(rnn, 1.0, 1.0, 2.0)
for i=1:rnn.ni
  @test rnn.b[i] == 0.0
end
for i=rnn.ni+1:rnn.n
  @test rnn.b[i] != 0.0
end

#
# test remove neuron
#
rnn1 = RNN_t(1,2,3)
rnn_modify_synapse!(rnn1, 1.1, 5.0, 10.0)
rnn_modify_bias!(rnn1, 1.0, 1.0, 2.0)
rnn2 = rnn_copy(rnn1)
rnn_remove_neuron!(rnn1, 1.1)
rnn_add_neuron!(rnn1, 1.0, 1.0, 2.0, 1.0, 20)
rnn_add_neuron!(rnn1, 1.0, 1.0, 2.0, 1.0, 20)
rnn_add_neuron!(rnn1, 1.0, 1.0, 2.0, 1.0, 20)
rnn_add_neuron!(rnn1, 1.0, 1.0, 2.0, 1.0, 20)
rnn_add_neuron!(rnn1, 1.0, 1.0, 2.0, 1.0, 20)
rnn_remove_neuron!(rnn1, 1.1)
rnn_remove_neuron!(rnn1, 1.1)
rnn_remove_neuron!(rnn1, 1.1)
@test rnn1.n          == 3
@test rnn1.ni         == 1
@test rnn1.no         == 2
@test rnn1.nh         == 0
@test size(rnn1.w)    == (3,3)
@test size(rnn1.a)    == (3,1)
@test size(rnn1.b)    == (3,1)
@test size(rnn1.o)    == (3,1)
@test rnn1.w[1:3,1:3] == rnn2.w[1:3,1:3]
@test rnn1.b[1:3]     == rnn2.b[1:3]

#
# test update function
#
rnn = RNN_t(1,2,0)
rnn.w[1,1] = 0.0
rnn.w[2,1] = 2.0
rnn.w[2,2] = 0.0
rnn.w[1,2] = 0.0
rnn.b[1]   = 1.0
rnn.b[2]   = 0.0
rnn.b[3]   = 0.0
rnn.a[1]   = 0.0
rnn.a[2]   = 0.0
rnn.a[3]   = 0.0
rnn.o[1]   = 0.0
rnn.o[2]   = 0.0
rnn.o[3]   = 0.0
rnn_update!(rnn)
@test rnn.b[1] == 1.0
@test rnn.b[2] == 0.0
@test rnn.b[3] == 0.0
@test rnn.a[1] == 1.0
@test rnn.a[2] == 0.0
@test rnn.a[3] == 0.0
@test rnn.o[1] == 1.0
@test rnn.o[2] == 0.0
@test rnn.o[3] == 0.0
rnn_update!(rnn)
@test rnn.b[1] == 1.0
@test rnn.b[2] == 0.0
@test rnn.b[3] == 0.0
@test rnn.a[1] == 1.0
@test rnn.a[2] == 2.0
@test rnn.a[3] == 0.0
@test rnn.o[1] == 1.0
@test rnn.o[2] == tanh(2.0)
@test rnn.o[3] == 0.0


ni1 = 20 + int(rand() + 50)
no1 = 20 + int(rand() + 50)
nh1 = 20 + int(rand() + 50)
ni2 = 20 + int(rand() + 50)
no2 = 20 + int(rand() + 50)
nh2 = 20 + int(rand() + 50)

rnn1   = RNN_t(ni1, no1, nh1)
rnn1.w = rand(size(rnn1.w))
rnn2   = RNN_t(ni2, no2, nh2)
rnn2.w = rand(size(rnn2.w))

rnn = rnn_merge_two(rnn1, rnn2)

@test rnn.ni == ni1 + ni2
@test rnn.no == no1 + no2
@test rnn.nh == nh1 + nh2

for src=1:rnn.n
  for dest=1:rnn.n

    # inputs to input neurons must all be zero
    if dest <= rnn.ni && src <= rnn1.ni + rnn2.ni
      @test rnn.w[dest, src] == 0.0
    end

    #
    # RNN 1 Tests
    #

    # rnn1 input to output
    if dest > rnn.ni && dest <= rnn.ni + rnn1.no && src <= rnn1.ni
      d = dest - rnn.ni + rnn1.ni
      @test rnn.w[dest, src] == rnn1.w[d, src]
    end

    # rnn1 input to hidden
    if dest > rnn.ni + rnn.no && dest <= rnn.ni + rnn.no + rnn1.nh && src <= rnn1.ni
      d = dest - rnn.ni + rnn1.ni - rnn.no + rnn1.no
      @test rnn.w[dest, src] == rnn1.w[d, src]
    end

    # rnn1 output to output
    if dest > rnn.ni && dest <= rnn.ni + rnn1.no &&
       src  > rnn.ni && src  <= rnn.ni + rnn1.no
      d = dest - rnn.ni + rnn1.ni
      s = src  - rnn.ni + rnn1.ni
      @test rnn.w[dest, src] == rnn1.w[d, s]
    end

    # rnn1 output to hidden
    if dest > rnn.ni + rnn.no && dest <= rnn.ni + rnn.no + rnn1.nh &&
       src  > rnn.ni          && src  <= rnn.ni + rnn1.no
      d = dest - rnn.ni + rnn1.ni - rnn.no + rnn1.no
      s = src  - rnn.ni + rnn1.ni
      @test rnn.w[dest, src] == rnn1.w[d, s]
    end

    # rnn1 hidden to output
    if dest > rnn.ni          && dest <= rnn.ni + rnn1.no &&
       src  > rnn.ni + rnn.no && src  <= rnn.ni + rnn.no + rnn1.nh
      d = dest - rnn.ni + rnn1.ni
      s = src  - rnn.ni + rnn1.ni - rnn.no + rnn1.no
      @test rnn.w[dest, src] == rnn1.w[d, s]
    end

    #= # rnn1 hidden to hidden =#
    if dest > rnn.ni + rnn.no && dest <= rnn.ni + rnn.no + rnn1.nh &&
       src  > rnn.ni + rnn.no && src  <= rnn.ni + rnn.no + rnn1.nh
      d = dest - rnn.ni - rnn.no + rnn1.ni + rnn1.no
      s = src  - rnn.ni - rnn.no + rnn1.ni + rnn1.no
      @test rnn.w[dest, src] == rnn1.w[d, s]
    end

    #
    # RNN 2 Tests
    #

    # rnn2 input to output
    if dest > rnn.ni + rnn1.no && dest <= rnn.ni + rnn.no &&
      src   > rnn1.ni          && src  <= rnn.ni
      d = dest - rnn.ni - rnn1.no + rnn2.ni
      s = src  - rnn1.ni
      @test rnn.w[dest, src] == rnn2.w[d, s]
    end

    #= # rnn2 input to hidden =#
    if dest >  rnn.ni + rnn.no + rnn1.nh &&
       dest <= rnn.ni + rnn.no + rnn.nh  &&
       src  >  rnn1.ni &&
       src  <= rnn.ni
      d = dest - rnn.ni - rnn.no - rnn1.nh + rnn2.ni + rnn2.no
      s = src  - rnn1.ni
      @test rnn.w[dest, src] == rnn2.w[d, s]
    end

    # rnn2 output to output
    if dest > rnn.ni + rnn1.no && dest <= rnn.ni + rnn.no &&
       src  > rnn.ni + rnn1.no && src  <= rnn.ni + rnn.no
      d = dest - rnn.ni - rnn1.no + rnn2.ni
      s = src  - rnn.ni - rnn1.no + rnn2.ni
      @test rnn.w[dest, src] == rnn2.w[d, s]
    end

    # rnn2 output to hidden
    if dest > rnn.ni + rnn.no + rnn1.nh && dest <= rnn.ni + rnn.no + rnn.nh &&
       src  > rnn.ni + rnn1.no          && src  <= rnn.ni + rnn.no
      d = dest - rnn.ni - rnn.no - rnn1.nh + rnn2.ni + rnn2.no
      s = src  - rnn.ni - rnn1.no + rnn2.ni
      @test rnn.w[dest, src] == rnn2.w[d, s]
    end

    # rnn2 hidden to output
    if dest > rnn.ni + rnn1.no && dest <= rnn.ni + rnn.no &&
       src  > rnn.ni + rnn.no + rnn1.nh && src  <= rnn.ni + rnn.no + rnn.nh
      d = dest - rnn.ni - rnn1.no + rnn2.ni
      s = src  - rnn.ni - rnn.no - rnn1.nh + rnn2.ni + rnn2.no
      @test rnn.w[dest, src] == rnn2.w[d, s]
    end

    # rnn2 hidden to hidden
    if dest > rnn.ni + rnn.no + rnn1.nh && dest <= rnn.ni + rnn.no + rnn.nh &&
       src  > rnn.ni + rnn.no + rnn1.nh && src  <= rnn.ni + rnn.no + rnn.nh
      d = dest - rnn.ni - rnn.no - rnn1.nh + rnn2.ni + rnn2.no
      s = src  - rnn.ni - rnn.no - rnn1.nh + rnn2.ni + rnn2.no
      @test rnn.w[dest, src] == rnn2.w[d, s]
    end



  end
end


rnn   = RNN_t(5, 5, 10)
rnn.w = ones(size(rnn.w))
rnn.b = ones(size(rnn.b)) .* 2.0

rnn_add_input_neurons!(rnn,3)
@test rnn.ni == 8
@test rnn.no == 5
@test rnn.nh == 10
@test rnn.n  == 23
for i=1:23
  if i in [6,7,8]
    @test rnn.b[i] == 0.0
  else
    @test rnn.b[i] == 2.0
  end
end

for i=1:23
  for j=1:23
    if i in [6:8]
      @test rnn.w[i,j] == 0.0
    elseif j in [6:8]
      @test rnn.w[i,j] == 0.0
    else
      @test rnn.w[i,j] == 1.0
    end
  end
end


rnn_add_output_neurons!(rnn,3)
@test rnn.ni == 8
@test rnn.no == 8
@test rnn.nh == 10
@test rnn.n  == 26
for i=1:26
  if i in [6:8]
    @test rnn.b[i] == 0.0
  elseif i in [14:16]
    @test rnn.b[i] == 0.0
  else
    @test rnn.b[i] == 2.0
  end
end

for i=1:26
  for j=1:26
    if i in [6:8]
      @test rnn.w[i,j] == 0.0
    elseif i in [14:16]
      @test rnn.w[i,j] == 0.0
    elseif j in [6:8]
      @test rnn.w[i,j] == 0.0
    elseif j in [14:16]
      @test rnn.w[i,j] == 0.0
    else
      @test rnn.w[i,j] == 1.0
    end
  end
end

rnn_add_hidden_neurons!(rnn,5)
@test rnn.ni == 8
@test rnn.no == 8
@test rnn.nh == 15
@test rnn.n  == 31
@test size(rnn.w) == (31,31)
@test size(rnn.b) == (31,1)
@test size(rnn.o) == (31,1)
@test size(rnn.a) == (31,1)

for i=1:31
  if i in [6:8]
    @test rnn.b[i] == 0.0
  elseif i in [14:16]
    @test rnn.b[i] == 0.0
  elseif i in [27:31]
    @test rnn.b[i] == 0.0
  else
    @test rnn.b[i] == 2.0
  end
end

for i=1:26
  for j=1:26
    if i in [6:8]
      @test rnn.w[i,j] == 0.0
    elseif i in [14:16]
      @test rnn.w[i,j] == 0.0
    elseif j in [6:8]
      @test rnn.w[i,j] == 0.0
    elseif j in [14:16]
      @test rnn.w[i,j] == 0.0
    elseif i in [27:31]
      @test rnn.w[i,j] == 0.0
    elseif j in [27:31]
      @test rnn.w[i,j] == 0.0
    else
      @test rnn.w[i,j] == 1.0
    end
  end
end

