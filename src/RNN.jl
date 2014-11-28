module RNN

export RNN_t
export rnn_modify_synapse!,     rnn_remove_synapse!, rnn_add_synapse!
export rnn_add_neuron!,         rnn_remove_neuron!,  rnn_modify_bias!
export rnn_copy,                rnn_read,            rnn_write
export rnn_merge_two
export rnn_add_input_neuron!,   rnn_add_output_neuron!
export rnn_add_input_neurons!,  rnn_add_output_neurons!
export rnn_add_hidden_neurons!, rnn_add_hidden_neuron!
export rnn_update!,             rnn_control!

type RNNInputSizeMismatch <: Exception
  expected::Int64
  received::Int64
end

showerror(io::IO, e::RNNInputSizeMismatch) = print(io, "RNN Input Mismatch: expected ", e.expected, " values, but received ", e.received)

type RNN_t
  ni::Int64
  no::Int64
  nh::Int64
  n::Int64

  w::Matrix{Float64}
  b::Matrix{Float64}
  o::Matrix{Float64}
  a::Matrix{Float64}

  RNN_t(ni::Int64, no::Int64, nh::Int64) =
    new(ni, no, nh, ni+no+nh,
        zeros(ni+no+nh, ni+no+nh), zeros(ni+no+nh,1),
        zeros(ni+no+nh,1),         zeros(ni+no+nh,1))
end

function rnn_add_input_neurons!(rnn::RNN_t, n::Integer)
  a = rnn.w[1:rnn.ni,:]
  b = rnn.w[rnn.ni+1:end,:]
  c = vcat(a,zeros(n,size(a)[2]),b)

  a = c[:,1:rnn.ni]
  b = c[:,rnn.ni+1:end]
  c = hcat(a,zeros(size(a)[1],n),b)

  rnn.w  = c
  rnn.b  = vcat(rnn.b[1:rnn.ni], zeros(n,1), rnn.b[rnn.ni+1:end])
  rnn.o  = vcat(rnn.o[1:rnn.ni], zeros(n,1), rnn.o[rnn.ni+1:end])
  rnn.a  = vcat(rnn.a[1:rnn.ni], zeros(n,1), rnn.a[rnn.ni+1:end])
  rnn.ni = rnn.ni + n
  rnn.n  = rnn.n  + n
  rnn
end

function rnn_add_input_neuron!(rnn::RNN_t, n::Integer)
  rnn_add_input_neurons!(rnn, 1)
end

function rnn_add_output_neurons!(rnn::RNN_t, n::Integer)
  a = rnn.w[1:rnn.ni+rnn.no,:]
  b = rnn.w[rnn.ni+rnn.no+1:end,:]
  c = vcat(a,zeros(n,size(a)[2]),b)

  a = c[:,1:rnn.ni+rnn.no]
  b = c[:,rnn.ni+rnn.no+1:end]
  c = hcat(a,zeros(size(a)[1],n),b)

  rnn.w  = c
  rnn.b  = vcat(rnn.b[1:rnn.ni+rnn.no], zeros(n,1), rnn.b[rnn.ni+rnn.no+1:end])
  rnn.a  = vcat(rnn.a[1:rnn.ni+rnn.no], zeros(n,1), rnn.a[rnn.ni+rnn.no+1:end])
  rnn.o  = vcat(rnn.o[1:rnn.ni+rnn.no], zeros(n,1), rnn.o[rnn.ni+rnn.no+1:end])
  rnn.no = rnn.no + n
  rnn.n  = rnn.n  + n
  rnn
end

function rnn_add_output_neuron!(rnn::RNN_t)
  rnn_add_output_neurons!(rnn, 1)
end

function rnn_add_hidden_neurons!(rnn::RNN_t, n::Integer)
  rnn.w  = vcat(rnn.w,zeros(n,size(rnn.w)[2]))
  rnn.w  = hcat(rnn.w,zeros(size(rnn.w)[1],n))
  rnn.b  = vcat(rnn.b, zeros(n,1))
  rnn.o  = vcat(rnn.o, zeros(n,1))
  rnn.a  = vcat(rnn.a, zeros(n,1))
  rnn.nh = rnn.nh + n
  rnn.n  = rnn.n  + n
  rnn
end

function rnn_add_hidden_neuron!(rnn::RNN_t)
  rnn_add_hidden_neurons!(rnn, 1)
end

function rnn_merge_two(rnn1, rnn2)
  ni  = rnn1.ni + rnn2.ni
  no  = rnn1.no + rnn2.no
  nh  = rnn1.nh + rnn2.nh
  rnn = RNN_t(ni, no, nh)

  # rnn1 input to output
  r1_srcs = 1:rnn1.ni
  r1_dest = rnn1.ni+1:rnn1.ni+rnn1.no
  r_srcs  = 1:rnn1.ni
  r_dest  = ni+1:ni+rnn1.no
  rnn.w[r_dest, r_srcs] = rnn1.w[r1_dest, r1_srcs]

  if rnn1.nh > 0
    # rnn1 input to hidden
    r1_srcs = 1:rnn1.ni
    r1_dest = rnn1.ni+rnn1.no+1:rnn1.ni+rnn1.no+rnn1.nh
    r_srcs  = 1:rnn1.ni
    r_dest  = ni+no+1:ni+no+rnn1.nh
    rnn.w[r_dest, r_srcs] = rnn1.w[r1_dest, r1_srcs]

    # rnn1 hidden to hidden
    r1_srcs = rnn1.ni+rnn1.no+1:rnn1.ni+rnn1.no+rnn1.nh
    r1_dest = r1_srcs
    r_srcs  = rnn.ni+rnn.no+1:rnn.ni+rnn.no+rnn1.nh
    r_dest  = r_srcs
    rnn.w[r_dest, r_srcs] = rnn1.w[r1_dest, r1_srcs]

    # rnn1 output to hidden
    r1_srcs = rnn1.ni+1:rnn1.ni+rnn1.no
    r1_dest = rnn1.ni+rnn1.no+1:rnn1.ni+rnn1.no+rnn1.nh
    r_srcs  = rnn.ni+1:rnn.ni+rnn1.no
    r_dest  = rnn.ni+rnn.no+1:rnn.ni+rnn.no+rnn1.nh
    rnn.w[r_dest, r_srcs] = rnn1.w[r1_dest, r1_srcs]

    # rnn1 hidden to output
    r1_srcs = rnn1.ni+rnn1.no+1:rnn1.ni+rnn1.no+rnn1.nh
    r1_dest = rnn1.ni+1:rnn1.ni+rnn1.no
    r_srcs  = rnn.ni+rnn.no+1:rnn.ni+rnn.no+rnn1.nh
    r_dest  = rnn.ni+1:rnn.ni+rnn1.no
    rnn.w[r_dest, r_srcs] = rnn1.w[r1_dest, r1_srcs]

  end

  #= # rnn1 output to output =#
  r1_srcs = rnn1.ni+1:rnn1.ni+rnn1.no
  r1_dest = r1_srcs
  r_srcs  = rnn.ni+1:rnn.ni+rnn1.no
  r_dest  = r_srcs
  rnn.w[r_dest, r_srcs] = rnn1.w[r1_dest, r1_srcs]


  #
  # RNN2
  #

  # rnn2 input to output
  r2_srcs = 1:rnn2.ni
  r2_dest = rnn2.ni+1:rnn2.ni+rnn2.no
  r_srcs  = rnn1.ni+1:ni
  r_dest  = ni+rnn1.no+1:ni+no
  rnn.w[r_dest, r_srcs] = rnn2.w[r2_dest, r2_srcs]


  if rnn2.nh > 0
    # rnn2 input to hidden
    r2_srcs = 1:rnn2.ni
    r2_dest = rnn2.ni+rnn2.no+1:rnn2.ni+rnn2.no+rnn2.nh
    r_srcs  = rnn1.ni+1:rnn.ni
    r_dest  = rnn.ni+rnn.no+rnn1.nh+1:rnn.ni+rnn.no+rnn.nh
    rnn.w[r_dest, r_srcs] = rnn2.w[r2_dest, r2_srcs]

    # rnn2 hidden to hidden
    r2_srcs = rnn2.ni+rnn2.no+1:rnn2.ni+rnn2.no+rnn2.nh
    r2_dest = r2_srcs
    r_srcs  = rnn.ni+rnn.no+rnn1.nh+1:rnn.ni+rnn.no+rnn.nh
    r_dest  = r_srcs
    rnn.w[r_dest, r_srcs] = rnn2.w[r2_dest, r2_srcs]

    # rnn2 output to hidden
    r2_srcs = rnn2.ni+1:rnn2.ni+rnn2.no
    r2_dest = rnn2.ni+rnn2.no+1:rnn2.ni+rnn2.no+rnn2.nh
    r_srcs  = rnn.ni+rnn1.no+1:rnn.ni+rnn.no
    r_dest  = rnn.ni+rnn.no+rnn1.nh+1:rnn.ni+rnn.no+rnn.nh
    rnn.w[r_dest, r_srcs] = rnn2.w[r2_dest, r2_srcs]

    # rnn2 hidden to output
    r2_srcs = rnn2.ni+rnn2.no+1:rnn2.ni+rnn2.no+rnn2.nh
    r2_dest = rnn2.ni+1:rnn2.ni+rnn2.no
    r_srcs  = rnn.ni+rnn.no+rnn1.nh+1:rnn.ni+rnn.no+rnn.nh
    r_dest  = rnn.ni+rnn1.no+1:rnn.ni+rnn.no
    rnn.w[r_dest, r_srcs] = rnn2.w[r2_dest, r2_srcs]

  end

  # rnn2 output to output
  r2_srcs = rnn2.ni+1:rnn2.ni+rnn2.no
  r2_dest = r2_srcs
  r_srcs  = rnn.ni+rnn1.no+1:rnn.ni+rnn.no
  r_dest  = r_srcs
  rnn.w[r_dest, r_srcs] = rnn2.w[r2_dest, r2_srcs]

  rnn.b[1:rnn1.ni]                        = rnn1.b[1:rnn1.ni]
  rnn.b[rnn1.ni+1:rnn.ni]                 = rnn2.b[1:rnn2.ni]
  rnn.b[rnn.ni+1:rnn.ni + rnn1.no]        = rnn1.b[rnn1.ni+1:rnn1.ni+rnn1.no]
  rnn.b[rnn.ni+rnn1.no+1:rnn.ni + rnn.no] = rnn2.b[rnn2.ni+1:rnn2.ni+rnn2.no]
  if rnn1.nh > 0
    rnn.b[rnn.ni+rnn.no+1:rnn.ni+rnn.no+rnn1.nh] = rnn1.b[rnn1.ni+rnn1.no+1:rnn1.n]
  end
  if rnn2.nh > 0
    rnn.b[rnn.ni+rnn.no+rnn1.nh+1:rnn.n]         = rnn2.b[rnn2.ni+rnn2.no+1:rnn2.n]
  end

  rnn
end

function rnn_write(rnn, output)
  f = open(output, "w")
  # write parameters
  @printf(f,"%d %d %d\n", rnn.ni, rnn.no, rnn.nh)
  # writing W
  for i in 1:rnn.n
    @printf(f,"%f",rnn.w[i,1])
    for j in 2:rnn.n
      @printf(f," %f",rnn.w[i,j])
    end
    @printf(f,"\n")
  end

  # writing b
  @printf(f,"%f",rnn.b[1])
  for n in 2:rnn.n
    @printf(f," %f",rnn.b[n])
  end
  @printf(f,"\n")
  close(f)
end


function rnn_read(input)
  f   = open(input, "r")
  sizes = split(strip(readline(f)))
  ni = int(eval(sizes[1]))
  no = int(eval(sizes[2]))
  nh = int(eval(sizes[3]))
  n = ni + no + nh
  rnn = RNN_t(ni, no, nh)
  # write parameters
  # writing W
  for i=1:n
    values = split(strip(readline(f)))
    for j=1:n
      rnn.w[i,j] = float64(eval(parse(values[j])))
    end
  end

  s = split(strip(readline(f)))
  for j=1:n
    rnn.b[j] = float64(eval(s[j]))
  end

  close(f)
  return rnn
end


function rnn_copy(rnn::RNN_t)
  new_rnn   = RNN_t(rnn.ni, rnn.no, rnn.nh)
  new_rnn.w = deepcopy(rnn.w)
  new_rnn.b = deepcopy(rnn.b)
  new_rnn.o = deepcopy(rnn.o)
  new_rnn.a = deepcopy(rnn.a)
  return new_rnn
end

function rnn_update!(rnn::RNN_t)
  rnn.a = rnn.b + rnn.w * rnn.o
  for i=1:rnn.ni
    rnn.o[i] = rnn.a[i]
  end
  for i=rnn.ni+1:rnn.n
    rnn.o[i]=tanh(rnn.a[i])
  end
end

#
# start - synapse functions
#
function rnn_remove_synapse!(rnn::RNN_t, prob::Float64)
  for i=1:rnn.n
    for j=1:rnn.n
      if rnn.w[i,j] != 0.0 && rand() <= prob
        rnn.w[i,j] = 0.0
      end
    end
  end
end

function rnn_add_synapse!(rnn::RNN_t, prob::Float64, delta::Float64)
  for i=rnn.ni+1:rnn.n
    for j=1:rnn.n
      if rnn.w[i,j] == 0.0 && rand() <= prob
        rnn.w[i,j] = (2.0 * rand() - 1.0) * delta
      end
    end
  end
end

function rnn_modify_synapse!(rnn::RNN_t, prob::Float64, delta::Float64, maximum::Float64)
  for i=rnn.ni+1:rnn.n
    for j=1:rnn.n
      if rnn.w[i,j] != 0.0
        if rand() <= prob
          rnn.w[i,j] += (2.0 * rand() - 1.0) * delta
          if abs(rnn.w[i,j]) > maximum
            if rnn.w[i,j] < 0
              rnn.w[i,j] = -maximum
            else
              rnn.w[i,j] =  maximum
            end
          end
        end
      end
    end
  end
end


#
# end - synapse functions
#

function rnn_modify_bias!(rnn::RNN_t, prob::Float64,
                                      delta::Float64,
                                      maximum::Float64)
  for i=rnn.ni+1:rnn.n
    if rand() <= prob
      rnn.b[i] = (2.0 * rand() - 1.0) * delta
      if abs(rnn.b[i]) > maximum
        if rnn.b[i] < 0
          rnn.b[i] = -maximum
        else
          rnn.b[i] =  maximum
        end
      end
    end
  end
end

function rnn_add_neuron!(rnn::RNN_t, prob::Float64,    delta_b::Float64,
                                     delta_w::Float64, prob_c::Float64,
                                     max_nh::Int64)
  if rnn.nh >= max_nh
    return
  end
  if rand() <= prob
    rnn.nh += 1
    rnn.n  += 1

    rnn.a        = cat(1, rnn.a, zeros(1))
    rnn.b        = cat(1, rnn.b, zeros(1))
    rnn.o        = cat(1, rnn.o, zeros(1))

    rnn.w        = cat(1, rnn.w, zeros(1,size(rnn.w)[2]))
    rnn.w        = cat(2, rnn.w, zeros(size(rnn.w)[1],1))

    rnn.b[rnn.n] = (2.0 * rand() - 1.0) * delta_b

    for i = rnn.ni+1:rnn.n
      if rand() <= prob_c
        rnn.w[i,rnn.n] = (2.0 * rand() - 1.0) * delta_w
      end
    end
    for i = 1:rnn.n
      if rand() <= prob_c
        rnn.w[rnn.n,i] = (2.0 * rand() - 1.0) * delta_w
      end
    end
  end
end

function rnn_remove_neuron!(rnn::RNN_t, prob::Float64)
  if rnn.nh > 0
    r = [1:rnn.n]
    for i=rnn.nh:-1:1
      if rand() <= prob
        splice!(r,rnn.ni+rnn.no+i)
      end
    end
    r = sort(r)
    rnn.w  = rnn.w[r,:]
    rnn.w  = rnn.w[:,r]
    rnn.a  = rnn.a[r,:]
    rnn.o  = rnn.o[r,:]
    rnn.b  = rnn.b[r,:]
    rnn.n  = length(r)
    rnn.nh = length(r) - rnn.ni - rnn.no
  end
end

function rnn_control!(rnn::RNN_t, input::Vector{Float64})
  if rnn.ni != length(input)
    error("RNN Input Mismatch: expected ", rnn.ni, " values, but received ", length(input))
    exit(-1)
  end
  for i=1:rnn.ni
    rnn.b[i] = input[i]
  end
  rnn_update!(rnn)
  r   = zeros(rnn.no)
  for i=1:rnn.no
    r[i]=rnn.o[rnn.ni+i]
  end
  return r
end
control!(rnn::RNN_t, input::Vector{Float64}) = rnn_control!(rnn, input)

end # module
