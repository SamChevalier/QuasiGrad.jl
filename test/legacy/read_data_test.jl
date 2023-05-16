# parse the jason into dicts - ac lines
function parse_json_acline_raw_params(json_data::Dict)
    acline_inds      = keys(json_data["network"]["ac_line"])
    aclines          = json_data["network"]["ac_line"]
    naclines         = length(acline_inds)
    r_series         = Float64.([aclines[ind]["r"] for ind in acline_inds])
    x_series         = Float64.([aclines[ind]["x"] for ind in acline_inds])
    b_ch             = Float64.([aclines[ind]["b"] for ind in acline_inds])
    from_bus         = parse.(Int64,[aclines[ind]["fr_bus"] for ind in acline_inds])
    to_bus           = parse.(Int64,[aclines[ind]["to_bus"] for ind in acline_inds])
    acline_ids       = [aclines[ind]["uid"] for ind in acline_inds]
    additional_shunt = Bool.([aclines[ind]["additional_shunt"] for ind in acline_inds]) # findfirst(x -> x.=="C34",line_id)

    # add additional shunts
    shunt_lines        = findall(x -> x == true, additional_shunt)
    g_fr               = zeros(Float64,naclines)
    b_fr               = zeros(Float64,naclines)
    g_to               = zeros(Float64,naclines)
    b_to               = zeros(Float64,naclines)
    g_fr_nonzero       = [aclines[ind]["g_fr"] for ind in shunt_lines]
    b_fr_nonzero       = [aclines[ind]["b_fr"] for ind in shunt_lines]
    g_to_nonzero       = [aclines[ind]["g_to"] for ind in shunt_lines]
    b_to_nonzero       = [aclines[ind]["b_to"] for ind in shunt_lines]
    g_fr[shunt_lines] .= g_fr_nonzero
    b_fr[shunt_lines] .= b_fr_nonzero
    g_to[shunt_lines] .= g_to_nonzero
    b_to[shunt_lines] .= b_to_nonzero

    # output
    return r_series, x_series, b_ch, g_fr, b_fr, g_to, b_to, from_bus, to_bus
end

# parse the jason into dicts - xfm
function parse_json_xfm_raw_params(json_data::Dict)
    xfm_inds         = keys(json_data["network"]["two_winding_transformer"])
    xfms             = json_data["network"]["two_winding_transformer"]
    nxfms            = length(xfm_inds)
    r_series         = Float64.([xfms[ind]["r"] for ind in xfm_inds])
    x_series         = Float64.([xfms[ind]["x"] for ind in xfm_inds])
    b_ch             = Float64.([xfms[ind]["b"] for ind in xfm_inds])
    from_bus         = parse.(Int64,[xfms[ind]["fr_bus"] for ind in xfm_inds])
    to_bus           = parse.(Int64,[xfms[ind]["to_bus"] for ind in xfm_inds])
    xfm_ids          = [xfms[ind]["uid"] for ind in xfm_inds]
    additional_shunt = Bool.([xfms[ind]["additional_shunt"] for ind in xfm_inds])

    # add additional shunts
    shunt_xfms         = findall(x -> x == true, additional_shunt)
    g_fr               = zeros(Float64,nxfms)
    b_fr               = zeros(Float64,nxfms)
    g_to               = zeros(Float64,nxfms)
    b_to               = zeros(Float64,nxfms)
    g_fr_nonzero       = [xfms[ind]["g_fr"] for ind in shunt_xfms]
    b_fr_nonzero       = [xfms[ind]["b_fr"] for ind in shunt_xfms]
    g_to_nonzero       = [xfms[ind]["g_to"] for ind in shunt_xfms]
    b_to_nonzero       = [xfms[ind]["b_to"] for ind in shunt_xfms]
    g_fr[shunt_xfms]  .= g_fr_nonzero
    b_fr[shunt_xfms]  .= b_fr_nonzero
    g_to[shunt_xfms]  .= g_to_nonzero
    b_to[shunt_xfms]  .= b_to_nonzero

    # output
    return r_series, x_series, b_ch, g_fr, b_fr, g_to, b_to, from_bus, to_bus
end