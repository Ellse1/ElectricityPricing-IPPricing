import sys
import pandas as pd
sys.path.insert(0,"..")
import gurobipy as gp
from gurobipy import GRB



# All keys of beta_b_bid_t_mw_mwsegm as numbers not string representation

def optimize(u_st_opt = None):


    # read data from the csv energy seller file, and the csv energy buyer file
    read_csv_data()


    try:
        
        # Create a new model
        gurobi_model = gp.Model("mip1")
        
        # add gurobi variables
        # variables for each buyer, for each period (how much power is the buyer buying in this period), continous, lb = 0
        x_bt = gurobi_model.addVars(buyer_id_list, periods, lb=0, vtype=GRB.CONTINUOUS, name='x_bt')
        # variables for each buyer, for each period, for each bid l (specific bid with MW and price)
        x_btl = dict()
        for l in beta_b_bid_t_mw_mwsegm.keys():
            x_btl[l] = gurobi_model.addVar(lb=0, vtype = GRB.CONTINUOUS, name='x_btl')

        
        # variables for each seller, for each period (how much power is the seller selling in this period), continous, lb = 0
        y_st = gurobi_model.addVars(seller_id_list, periods, lb=0, vtype=GRB.CONTINUOUS, name='y_st')
        # variables for each seller, for each period, for each offer (specific offer with MW and price)
        y_stl = dict()
        for l in beta_s_a_t_mw_mwsegm.keys():
            y_stl[l] = gurobi_model.addVar(lb=0, vtype = GRB.CONTINUOUS, name='y_stl')
        
        
        # seller comittment variable(binary): variables for each seller, for each period (is a seller comitted or not in a specific period)
        if(u_st_opt == None):
            u_st = gurobi_model.addVars(seller_id_list, periods, vtype=GRB.BINARY, name='u_st')
        else:
            u_st = gurobi_model.addVars(seller_id_list, periods, name='u_st')
            gurobi_model.addConstrs(u_st[s, t] == u_st_opt[s, t] for s in seller_id_list for t in periods)

        # start-up indicator as written in the example code: gurobi_dcopf.py 
        phi_st = gurobi_model.addVars(seller_id_list, periods, lb=0, ub=GRB.INFINITY, name='phi_st')

        # voltage angle alpha_vt and flow on the line connection f_vwt (as described in the example code: gurobi_dcopf.py) 
        # are not needed in this model 
        
        
        # Process any pending model modifications. 
        gurobi_model.update()
        
        
        # set objective function
        gurobi_model.setObjective(
            gp.quicksum(
                # v_btl [$/MWh]
                beta_b_bid_t_mw_mwsegm[l] *
                # x_btl [MWh]
                x_btl[l]
                for l in beta_b_bid_t_mw_mwsegm
            ) 
            -
            gp.quicksum(
                # c_stl [$/MWh]
                beta_s_a_t_mw_mwsegm[l] *
                # y_stl [MWh]
                y_stl[l]
                for l in beta_s_a_t_mw_mwsegm
            )
            -
            # no load cost of all bids from all sellers that are comitt
            gp.quicksum(
                # h_st [$/h] (no load price)
                no_load_cost *
                # u_st [0,1] (seller comittment variable)
                u_st[s, t]
    	        for s, t, no_load_cost in seller_data[["Masked Lead Participant ID", "Trading Interval", "No Load Price"]].itertuples(index=False)
            )
            , GRB.MAXIMIZE
        )
                
        # Process any pending model modifications. 
        gurobi_model.update()


        # add constraints

        # constraint 1:
        gurobi_model.addConstrs(x_btl[l] >= 0 for l in beta_b_bid_t_mw_mwsegm)
        # each power_consumption of a buyer is smaller or equal to the possible maximum power_consumption of the buyer in the specific mw segment
        gurobi_model.addConstrs(x_btl[(buyer, bid_id, t, mw, mw_segm)] <= mw for (buyer, bid_id, t, mw, mw_segm) in beta_b_bid_t_mw_mwsegm)

        # find all fixed power of the buyers in the specific period
        buyer_fixed_power_in_t = dict()
        # and find all maximum powers of the buyers in the specific period
        buyer_max_power_in_t = dict()
        for buyer_bid in buyer_data.iterrows():
            #1 fixed power
            # if not in dict jet
            if(buyer_fixed_power_in_t.get( (int(buyer_bid[1]["Masked Lead Participant ID"]), int(buyer_bid[1]["Hour"])) ) == None):
                buyer_fixed_power_in_t[( int(buyer_bid[1]["Masked Lead Participant ID"]), int(buyer_bid[1]["Hour"]) )] = 0
            # only sum up if FIXED
            if(buyer_bid[1]["Bid Type"] == "FIXED" and buyer_bid[1]["Segment 1 MW"] != None and buyer_bid[1]["Segment 1 MW"] != ""):
                buyer_fixed_power_in_t[(int(buyer_bid[1]["Masked Lead Participant ID"]), int(buyer_bid[1]["Hour"]))] += float(buyer_bid[1]["Segment 1 MW"])
            #2 maximum power
            # if not in dict jet
            if(buyer_max_power_in_t.get( (int(buyer_bid[1]["Masked Lead Participant ID"]), int(buyer_bid[1]["Hour"])) ) == None):
                buyer_max_power_in_t[( int(buyer_bid[1]["Masked Lead Participant ID"]), int(buyer_bid[1]["Hour"]) )] = 0
            # go through all MW segments of the bid 
            for mw_segment in range(1, 51):
                if(pd.isna(buyer_bid[1]["Segment " + str(mw_segment) + " MW"])):
                    break
                buyer_max_power_in_t[(int(buyer_bid[1]["Masked Lead Participant ID"]), int(buyer_bid[1]["Hour"]))] += float(buyer_bid[1]["Segment " + str(mw_segment) + " MW"])


        # constraint 2:
        # power of buyer b in period t - optional power = fixed power of buyer b in period t 
        gurobi_model.addConstrs(
            # power of buyer b in period t
            x_bt[buyer, timestamp] - 
            # optional power of buyer b in period t
            gp.quicksum(x_btl[(b, bid, t, m, ms)] for (b, bid, t, m, ms) in beta_b_bid_t_mw_mwsegm if b == buyer and t == timestamp) 
            == 
            # fixed power of buyer b in period t
            buyer_fixed_power_in_t[(buyer, timestamp)]
            # for... 
            for (buyer, bid_id, timestamp, megaw, megaw_segm) in beta_b_bid_t_mw_mwsegm
        )

        # power of buyer b in period t is >= fixed power of buyer b in period t
        gurobi_model.addConstrs(x_bt[buyer, t] >= buyer_fixed_power_in_t[(buyer, t)] for buyer in buyer_id_list for t in periods)
        
        # constraint 3:
        # x_bt is smaller than max Power in this period
        # x_bt is smaller than sum of all power segments of all bids of the buyer in the specific period
        gurobi_model.addConstrs(x_bt[buyer, t] <= buyer_max_power_in_t[(buyer, t)] for buyer in buyer_id_list for t in periods)        


        # constraint 4:
        gurobi_model.addConstrs(y_stl[l] >= 0 for l in beta_s_a_t_mw_mwsegm)

        # constraint 5:
        # power of seller s in period t for bid l is less than the maximum power possible for this bid
        # y_stl - q_stl * u_st <= 0s
        gurobi_model.addConstrs(y_stl[(s, a, t, mw, mw_set)] - mw * u_st[s, t] <= 0 for (s, a, t, mw, mw_set) in beta_s_a_t_mw_mwsegm)


        
        # find all maximum powers of the sellers in the specific period
        seller_max_power_in_t = dict()
        # and the maximum power of the specific bid of a seller in the specific period (sum up all mw segments of the specific bid-row)
        # seller_max_power_in_bid_in_t = dict()
        # and the must_run_powers of the seller of the specific bid
        seller_must_run_power_in_bid_in_t = dict()
        # and all must_run_powers of the sellers in the specific period (sum up all must_run_powers of the sellers in the specific period)
        seller_must_run_power_in_t = dict()
        for seller_offer in seller_data.iterrows():
            #1 maximum power
            # if not in dict jet
            if(seller_max_power_in_t.get( (int(seller_offer[1]["Masked Lead Participant ID"]), int(seller_offer[1]["Trading Interval"])) ) == None):
                seller_max_power_in_t[( int(seller_offer[1]["Masked Lead Participant ID"]), int(seller_offer[1]["Trading Interval"]) )] = 0
            # go through all MW segments of the offer 
            for mw_segment in range(1, 11):
                if(pd.isna(seller_offer[1]["Segment " + str(mw_segment) + " MW"])):
                    break
                seller_max_power_in_t[(int(seller_offer[1]["Masked Lead Participant ID"]), int(seller_offer[1]["Trading Interval"]))] += float(seller_offer[1]["Segment " + str(mw_segment) + " MW"])
            #2 maximum power of the specific bid
            # seller_max_power_in_bid_in_t[( int(seller_offer[1]["Masked Lead Participant ID"]), int(seller_offer[1]["Masked Asset ID"]), int(seller_offer[1]["Trading Interval"]) )] = 0
            # go through all MW segments of the offer
            # for mw_segment in range(1, 11):
                # if(pd.isna(seller_offer[1]["Segment " + str(mw_segment) + " MW"])):
                #     break
                # seller_max_power_in_bid_in_t[(int(seller_offer[1]["Masked Lead Participant ID"]), int(seller_offer[1]["Masked Asset ID"]), int(seller_offer[1]["Trading Interval"]))] += float(seller_offer[1]["Segment " + str(mw_segment) + " MW"])
            
            #3 must_run_power of the specific bid
            seller_must_run_power_in_bid_in_t[(int(seller_offer[1]["Masked Lead Participant ID"]), int(seller_offer[1]["Masked Asset ID"]), int(seller_offer[1]["Trading Interval"]))] = float(seller_offer[1]["Must Take Energy"])

            #4 must_run_power of the seller in the specific period
            # if not in dict jet
            if(seller_must_run_power_in_t.get( (int(seller_offer[1]["Masked Lead Participant ID"]), int(seller_offer[1]["Trading Interval"])) ) == None):
                seller_must_run_power_in_t[( int(seller_offer[1]["Masked Lead Participant ID"]), int(seller_offer[1]["Trading Interval"]) )] = 0
            # only sum up if MUST_RUN
            if(seller_offer[1]["Unit Status"] == "MUST_RUN"):
                seller_must_run_power_in_t[(int(seller_offer[1]["Masked Lead Participant ID"]), int(seller_offer[1]["Trading Interval"]))] += float(seller_offer[1]["Must Take Energy"])


        # constraint 6:
        # y_st - (sum of y_stl  with same s and t) == 0
        # gurobi_model.addConstrs(y_st[s, t] - gp.quicksum(y_stl[l] for l in beta_s_a_t_mw_mwsegm if l[3] == s and l[2] == t) == 0 for s in seller_id_list for t in periods)
        gurobi_model.addConstrs((y_st[seller, time] == 
                                 gp.quicksum(y_stl[(seller, a, time, m, ms)] for (s, a, t, m, ms) in beta_s_a_t_mw_mwsegm if s == seller and t == time )) 
                            for (seller, asset, time, mw, mw_set) in beta_s_a_t_mw_mwsegm)

        # constraint 7:
        # y_st - must_run_power of period * comitted >= 0
        # power production by seller s in period t - must_run_power of seller s in period t * comittment of seller s in period t >= 0
        gurobi_model.addConstrs( (y_st[s, t] - seller_must_run_power_in_t[(s, t)] * u_st[s, t] >= 0 ) for s in seller_id_list for t in periods)
        
        '''
        # Do, when there are bids with must-run-power
        # maby constraint needed: for a specific bid, the must-run power needs to be considered
        gurobi_model.addConstrs( gp.quicksum(
            y_stl[(s, a, t, mw, mw_seg)] for (s, a, t, mw, mw_seg) in beta_s_a_t_mw_mwsegm if s == seller and a == asset and t == t
        ) >= seller_must_run_power_in_bid_in_t[(seller, asset, t)] for (seller, asset, t) in seller_must_run_power_in_bid_in_t)
        '''


        # constraint 8:
        # y_stl - u_st * P_st <= 0
        # power production by offer l - comittment of seller s * maximum power production of seller s in period t <= 0
        gurobi_model.addConstrs( (y_stl[(s, a, t, mw, mw_segm)] - u_st[s, t] * seller_max_power_in_t[(s, t)] <= 0 )
                                for (s, a, t, mw, mw_segm) in beta_s_a_t_mw_mwsegm
        )

        # - constraint 9:
        # phi_st - u_st + u_{st-1} >= 0
        # it does't work, that the seller did not start now, is used now but was not used in the period before
        gurobi_model.addConstrs(phi_st[s, t] - u_st[s, t] + u_st[s, t-1] >= 0 for s in seller_id_list for t in periods[1:])


        # constraint 10:
        # changed: not uptime but maximum-daily-energy is restricted for each bid
        # ignored for now


        # constraint 13:
        # production is equal to consumption in every period
        gurobi_model.addConstrs(((gp.quicksum(x_bt[buyer, t] for buyer in buyer_id_list) == gp.quicksum(y_st[seller, t] for seller in seller_id_list)) for t in periods), name = "demand_supply_balance")


        # process any pending model modifications
        gurobi_model.update()
           
        # Optimize model
        gurobi_model.optimize()

        # for v in gurobi_model.getVars():
        #    print('%s %g' % (v.VarName, v.X))

        status = gurobi_model.getAttr('Status')
        print("Status: " + str(status))
        print("u_st_opt equals none? " + str(u_st_opt == None))
        if (status == 2 and u_st_opt == None):
            print("Ready for second run")
            # do the optimization again without binary variables
            u_st_opt = dict()
            for s in seller_id_list:
                for t in periods:
                    u_st_opt[s, t] = u_st[s, t].X
            
            # dispose the model and the environment (to create new one in recursive call)
            gurobi_model.dispose()
            gp.disposeDefaultEnv()
            # recursive call
            optimize(u_st_opt)
        
        # print variables and shadow prices
        # this is the very end status, when the optimising was successful and shaddow prices are available
        elif (status == GRB.OPTIMAL and u_st_opt != None):
            # for v in gurobi_model.getVars():
                # print('%s %g' % (v.VarName, v.X))
            
            print("Optimal objective: %g" % gurobi_model.objVal)
            print()
            # print("Shadow prices:")
            shadow_prices = dict()
            for p in periods:
                c_p = gurobi_model.getConstrByName(f'demand_supply_balance[{p}]')
                # get shadow prices for each node for each period
                shadow_prices[p] = abs(c_p.Pi)
                print(f"Period {p} price: {shadow_prices[p]}")
            
            # store all the important data in data dicts
            # store for every asset_id, for every period: (energy produced, asked_earnings, shadow_price_based_earnings, loss_or_earning)
            seller_asset_energy_produced = dict()
            seller_asset_asked_earnings = dict()
            seller_asset_price_based_earnings = dict()
            seller_asset_start_up_costs = dict()
            #start up indicator to calculate than the start up costs
            asset_startup_indicator = dict()

            for p in periods:
                for asset_id in seller_asset_id_list:
                    for (s, a, t, mw, mwseg) in beta_s_a_t_mw_mwsegm:
                        if(a == asset_id and t == p):
                            # energy produced
                            if(seller_asset_energy_produced.get((asset_id, p)) == None):
                                seller_asset_energy_produced[(asset_id, p)] = 0
                            seller_asset_energy_produced[(asset_id, p)] += y_stl[(s, a, t, mw, mwseg)].X
                            # asked_earnings
                            if(seller_asset_asked_earnings.get((asset_id, p)) == None):
                                seller_asset_asked_earnings[(asset_id, p)] = 0.0 
                            seller_asset_asked_earnings[(asset_id, p)] += y_stl[(s, a, t, mw, mwseg)].X * float(beta_s_a_t_mw_mwsegm[(s, a, t, mw, mwseg)])
                            # price based earnings
                            if(seller_asset_price_based_earnings.get((asset_id, p)) == None):
                                seller_asset_price_based_earnings[(asset_id, p)] = 0.0
                            seller_asset_price_based_earnings[(asset_id, p)] += y_stl[(s, a, t, mw, mwseg)].X * float(shadow_prices[p])

                            # fill the startup indicator for the asset
                            if(p > 1):
                                try:
                                    if(int(y_stl[(s, a, t-1, mw, mwseg)].X) == 0 and int(y_stl[(s, a, t, mw, mwseg)].X) > 0):
                                    #if(int(y_stl[(s, a, t, mw, mwseg)].X) > 0):
                                        asset_startup_indicator[(asset_id, p)] = 1
                                    else:
                                        asset_startup_indicator[(asset_id, p)] = 0
                                # could throw a key error, if t-1 does not exist
                                except KeyError:
                                    if(int(y_stl[(s, a, t, mw, mwseg)].X) > 0):
                                        asset_startup_indicator[(asset_id, p)] = 1
                                    else:
                                        asset_startup_indicator[(asset_id, p)] = 0
                            else:
                                if(int(y_stl[(s, a, t, mw, mwseg)].X) > 0):
                                    asset_startup_indicator[(asset_id, p)] = 1
                                else:
                                    asset_startup_indicator[(asset_id, p)] = 0
                                    


                            

                    # start up costs
                    
                    if(seller_asset_start_up_costs.get((asset_id, p)) == None):
                        seller_asset_start_up_costs[(asset_id, p)] = 0.0
                    try:    
                        start_up_price = seller_data.loc[(seller_data["Masked Asset ID"] == str(asset_id)) & (seller_data["Trading Interval"] == p)].iloc[0].get("Cold Startup Price")
                        seller_asset_start_up_costs[(asset_id, p)] = float(start_up_price) * float(asset_startup_indicator[(asset_id, p)])
                    # could be, that for one asset one period is not defined... 
                    except Exception:
                        start_up_price = 0.0
                        seller_asset_start_up_costs[(asset_id, p)] = 0.0
                        print("one exception for asset_id: " + str(asset_id) + " and period: " + str(p))
                    
            '''
            print("Optimization finished")
            print(seller_asset_energy_produced)
            print(seller_asset_asked_earnings)
            print(seller_asset_price_based_earnings)
            '''
            print(asset_startup_indicator)
            print(seller_asset_start_up_costs)

        # dispose the model and the environment (to create new one in recursive call)
        gurobi_model.dispose()
        gp.disposeDefaultEnv()

        return 
        
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
        gurobi_model.dispose()
        gp.disposeDefaultEnv()

    except AttributeError:
        print('Encountered an attribute error')
        gurobi_model.dispose()
        gp.disposeDefaultEnv()

    except Exception:
        print('Encountered an exception: ' + str(sys.exc_info()[0]))
        gurobi_model.dispose()
        gp.disposeDefaultEnv()
    
    

def read_csv_data():
    # READ THE DATA FROM SELLERS FILE
    # read data from the csv energy offers file, and skip the first 6 rows (header)
    global seller_data
    # seller_data = pd.read_csv("./data/PreparedSellerData.csv", sep=";", skiprows=4)
    seller_data = pd.read_csv("./data/test2-PreparedSellerData.csv", sep=";", skiprows=4)

    # remove the first row and the last row (unit of every entry and number of all offers)
    seller_data = seller_data.iloc[1:-1, :]

    seller_data["Masked Lead Participant ID"] = seller_data["Masked Lead Participant ID"].astype(int)
    seller_data["Trading Interval"] = seller_data["Trading Interval"].astype(int)

    # create a list of all the sellers (soller_ids)
    global seller_id_list
    seller_id_list = seller_data["Masked Lead Participant ID"].unique().tolist()
    seller_id_list = sorted(set([int(x) for x in seller_id_list]))

    global seller_asset_id_list
    seller_asset_id_list = seller_data["Masked Asset ID"].unique().tolist()
    seller_asset_id_list = sorted(set([int(x) for x in seller_asset_id_list]))


    # READ THE DATA FROM BUYERS FILEs
    # read data from the csv energy buyer file, and skip the first 4 rows (header)
    global buyer_data
    # buyer_data = pd.read_csv("./data/PreparedBuyerData.csv", sep=";", skiprows=4)
    buyer_data = pd.read_csv("./data/test2-PreparedBuyerData.csv", sep=";", skiprows=4)
    buyer_data = buyer_data.dropna(axis=1, how='all')

    # remove the first row and the last row (unit of every entry and number of all offers)
    buyer_data = buyer_data.iloc[1:-1, :]
    buyer_data["Masked Lead Participant ID"] = buyer_data["Masked Lead Participant ID"].astype(int)
    buyer_data["Hour"] = buyer_data["Hour"].astype(int)
    # create a list of all the buyers (buyer_ids)
    global buyer_id_list
    buyer_id_list = buyer_data["Masked Lead Participant ID"].unique().tolist()
    buyer_id_list = set([int(x) for x in buyer_id_list])

    
    # READ THE PERIOD DATA FROM THE BUYER FILE (1 - 24)
    global periods
    periods = sorted(set([int(x) for x in buyer_data["Hour"].unique().tolist()]))
        
    # Create the beta_b_bid_t_mw_mwsegm list from buyers, 
    # list of (Bidder, Bid_id, t, MW MWsegment) -> price   pairs for each buyer, for each period step (also with multiple MW blocks from one BID)
    # MWssegment is needed, becaus it can be, that a buyer pays for first 200mw 10$/MWh and for the next 200MW 8$/MWh -> Same MW number
    global beta_b_bid_t_mw_mwsegm
    beta_b_bid_t_mw_mwsegm = dict()

    # go through the 24 periods
    for t in periods:
        # go through all buyers
        for b in buyer_id_list:
            # go through all bids of the buyer in this period (bid_id) -> now i have one row
            for bid in buyer_data[(buyer_data['Bid Type'] == "PRICE") & (buyer_data['Masked Lead Participant ID'] == b) & (buyer_data['Hour'] == t)].iterrows():
                # go through every MW step of this bid (there can be 50 MW steps)
                for mw_segment in range(1, 51):
                    if(pd.isna(bid[1]["Segment " + str(mw_segment) + " MW"])):
                        break

                    beta_b_bid_t_mw_mwsegm[(b, int(bid[1]["Bid ID"]), t, float(bid[1]["Segment " + str(mw_segment) + " MW"]), mw_segment)] =  bid[1]["Segment " + str(mw_segment) + " Price"]

    print("beta_b_bid_t_mw_mwseg filled")
    print("Number of (Buyer, bid, t, mw, mwseg) pairs: " + str(len(beta_b_bid_t_mw_mwsegm)))
    print()
    
    # Create the beta_s_t_mw_mwsegm
    global beta_s_a_t_mw_mwsegm
    beta_s_a_t_mw_mwsegm = dict()
    
    # go through the 24 periods
    for t in periods:
        # go through all sellers
        for s in seller_id_list:
            # go through all offers of the seller in this period (with ECONMIC BID TYPE (Unit status)) 
            # for offer in seller_data[(seller_data["Masked Lead Participant ID"] == s) & (seller_data["Trading Interval"] == t) & (seller_data["Unit Status"] == "ECONOMIC")].iterrows():
            for offer in seller_data[(seller_data["Masked Lead Participant ID"] == s) & (seller_data["Trading Interval"] == t)].iterrows():
                # go through every MW step of this offer (there can be 10 MW steps)
                for mw_segment in range(1, 11):
                    if(pd.isna(offer[1]["Segment " + str(mw_segment) + " MW"])):
                        break
                    beta_s_a_t_mw_mwsegm[(s, int(offer[1]["Masked Asset ID"]), t, float(offer[1]["Segment " + str(mw_segment) + " MW"]), mw_segment)] =  offer[1]["Segment " + str(mw_segment) + " Price"]
                
    print("beta_s_t_mw_mwseg filled")
    print("Number of (Seller, assed_id, Timestep, MW, MW_segm) pairs: " + str(len(beta_s_a_t_mw_mwsegm)))
    print()

# Start the main app
if __name__ == "__main__":
    optimize()
