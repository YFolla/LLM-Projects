import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from amadeus import Client

load_dotenv()

MODEL = 'gpt-4o-mini'
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load Amadeus credentials
AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")

# Initialize Amadeus client
if not AMADEUS_API_KEY or not AMADEUS_API_SECRET:
    print("⚠️  WARNING: Amadeus API credentials not found in .env file!")
    print("Please ensure AMADEUS_API_KEY and AMADEUS_API_SECRET are set in your .env file")
    amadeus = None
else:
    try:
        amadeus = Client(
            client_id=AMADEUS_API_KEY,
            client_secret=AMADEUS_API_SECRET
        )
        print("✅ Amadeus SDK initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Amadeus SDK: {e}")
        amadeus = None

system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers. "
system_message += "Always be accurate. When users ask about flights, use the flight search tool to provide real-time information."

def get_flight_offers(origin, destination, departure_date, adults, 
                     return_date=None, children=None, infants=None, 
                     travel_class=None, non_stop=None, currency_code=None, 
                     max_price=None, max_results=5):
    """
    Search for flight offers using the Amadeus SDK.
    
    Required parameters:
    - origin: IATA airport/city code (e.g., 'JFK', 'NYC')
    - destination: IATA airport/city code (e.g., 'LHR', 'LON') 
    - departure_date: Date in YYYY-MM-DD format
    - adults: Number of adult passengers (1-9)
    
    Optional parameters:
    - return_date: Return date in YYYY-MM-DD format for round trips
    - children: Number of children (age 2-11)
    - infants: Number of infants (under 2)
    - travel_class: ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST
    - non_stop: True for non-stop flights only
    - currency_code: ISO currency code (e.g., 'USD', 'EUR')
    - max_price: Maximum price per traveler
    - max_results: Maximum number of offers to return (default: 5)
    """
    
    # Check if Amadeus client is available
    if not amadeus:
        return {"error": "Flight search service is currently unavailable. Please try again later."}
    
    # Validate required parameters
    if not origin or not isinstance(origin, str) or len(origin.strip()) == 0:
        return {"error": "Origin airport/city code is required and must be a valid string (e.g., 'JFK', 'NYC')"}
    
    if not destination or not isinstance(destination, str) or len(destination.strip()) == 0:
        return {"error": "Destination airport/city code is required and must be a valid string (e.g., 'LHR', 'LON')"}
    
    if not departure_date or not isinstance(departure_date, str):
        return {"error": "Departure date is required and must be in YYYY-MM-DD format (e.g., '2024-12-25')"}
    
    if not adults or not isinstance(adults, int) or adults < 1 or adults > 9:
        return {"error": "Adults parameter is required and must be an integer between 1 and 9"}
    
    # Validate date format (basic check)
    try:
        from datetime import datetime
        datetime.strptime(departure_date, '%Y-%m-%d')
        if return_date:
            datetime.strptime(return_date, '%Y-%m-%d')
    except ValueError:
        return {"error": "Date format must be YYYY-MM-DD (e.g., '2024-12-25')"}
    
    try:
        # Prepare parameters for the Amadeus SDK call
        search_params = {
            'originLocationCode': origin.strip().upper(),
            'destinationLocationCode': destination.strip().upper(),
            'departureDate': departure_date,
            'adults': adults,
            'max': min(max_results * 3, 50)  # Get more results to sort and filter
        }
        
        # Add optional parameters if provided
        if return_date:
            search_params['returnDate'] = return_date
        if children is not None:
            search_params['children'] = children
        if infants is not None:
            search_params['infants'] = infants
        if travel_class:
            search_params['travelClass'] = travel_class.upper()
        if non_stop is not None:
            search_params['nonStop'] = str(non_stop).lower()
        if currency_code:
            search_params['currencyCode'] = currency_code.upper()
        if max_price is not None:
            search_params['maxPrice'] = max_price
            
        print(f"🔍 DEBUG: Calling Amadeus SDK with params: {search_params}")
        
        # Make the Amadeus API call
        response = amadeus.shopping.flight_offers_search.get(**search_params)
        
        print(f"✅ DEBUG: Amadeus API call successful, found {len(response.data)} offers")
        
        if not response.data:
            return {"error": "No flights found for your search criteria. Try different dates or destinations."}
        
        # Process and filter flight offers
        processed_offers = []
        
        for offer in response.data:
            try:
                # Extract essential flight information
                price_info = offer.get('price', {})
                total_price = float(price_info.get('total', 0))
                currency = price_info.get('currency', 'EUR')
                
                # Get itinerary information
                itineraries = offer.get('itineraries', [])
                if not itineraries:
                    continue
                
                flight_info = {
                    'id': offer.get('id', ''),
                    'price': {
                        'total': total_price,
                        'currency': currency
                    },
                    'itineraries': []
                }
                
                for itinerary in itineraries:
                    segments = itinerary.get('segments', [])
                    if not segments:
                        continue
                    
                    itinerary_info = {
                        'duration': itinerary.get('duration', ''),
                        'segments': []
                    }
                    
                    for segment in segments:
                        departure = segment.get('departure', {})
                        arrival = segment.get('arrival', {})
                        carrier = segment.get('carrierCode', '')
                        flight_number = segment.get('number', '')
                        
                        segment_info = {
                            'departure': {
                                'airport': departure.get('iataCode', ''),
                                'time': departure.get('at', '')
                            },
                            'arrival': {
                                'airport': arrival.get('iataCode', ''),
                                'time': arrival.get('at', '')
                            },
                            'airline': carrier,
                            'flight_number': f"{carrier}{flight_number}",
                            'duration': segment.get('duration', '')
                        }
                        itinerary_info['segments'].append(segment_info)
                    
                    flight_info['itineraries'].append(itinerary_info)
                
                processed_offers.append(flight_info)
                
            except Exception as e:
                print(f"⚠️  Warning: Error processing offer {offer.get('id', 'unknown')}: {e}")
                continue
        
        # Sort by price (cheapest first)
        processed_offers.sort(key=lambda x: x['price']['total'])
        
        # Limit to requested number of results
        final_offers = processed_offers[:max_results]
        
        print(f"✅ DEBUG: Processed and filtered to {len(final_offers)} offers")
        
        # Return the processed flight offers
        return {
            "success": True,
            "flights": final_offers,
            "count": len(final_offers),
            "search_params": {
                "origin": origin.upper(),
                "destination": destination.upper(),
                "departure_date": departure_date,
                "return_date": return_date,
                "adults": adults,
                "children": children,
                "infants": infants,
                "travel_class": travel_class
            }
        }
        
    except Exception as e:
        print(f"❌ DEBUG: Amadeus SDK error: {e}")
        error_msg = str(e).lower()
        
        if "rate limit" in error_msg or "quota" in error_msg:
            return {"error": "The assistant is currently experiencing API rate limits. Please try again in a few moments."}
        elif "authentication" in error_msg or "unauthorized" in error_msg:
            return {"error": "The assistant is unable to obtain live flight data right now due to authentication issues."}
        elif "not found" in error_msg or "invalid" in error_msg:
            return {"error": "Please check your airport codes and dates. Make sure they are valid IATA codes and future dates."}
        elif "connection" in error_msg or "timeout" in error_msg:
            return {"error": "The assistant is unable to obtain live flight data right now due to connection issues."}
        else:
            return {"error": f"Unable to retrieve flight data: {str(e)}"}

# Tool definition for OpenAI function calling
flight_search_tool = {
    "type": "function",
    "function": {
        "name": "get_flight_offers",
        "description": "Search for flight offers using real-time data from Amadeus",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "Origin airport or city IATA code (e.g., 'JFK', 'NYC')"
                },
                "destination": {
                    "type": "string", 
                    "description": "Destination airport or city IATA code (e.g., 'LHR', 'LON')"
                },
                "departure_date": {
                    "type": "string",
                    "description": "Departure date in YYYY-MM-DD format"
                },
                "adults": {
                    "type": "integer",
                    "description": "Number of adult passengers (1-9)"
                },
                "return_date": {
                    "type": "string",
                    "description": "Return date in YYYY-MM-DD format for round trips"
                },
                "children": {
                    "type": "integer", 
                    "description": "Number of children (age 2-11)"
                },
                "infants": {
                    "type": "integer",
                    "description": "Number of infants (under 2)"
                },
                "travel_class": {
                    "type": "string",
                    "enum": ["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"],
                    "description": "Travel class preference"
                },
                "non_stop": {
                    "type": "boolean",
                    "description": "True to search only non-stop flights"
                },
                "currency_code": {
                    "type": "string",
                    "description": "ISO currency code (e.g., 'USD', 'EUR')"
                },
                "max_price": {
                    "type": "integer",
                    "description": "Maximum price per traveler"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of flight offers to return (default: 5)"
                }
            },
            "required": ["origin", "destination", "departure_date", "adults"]
        }
    }
}

# List of available tools
tools = [flight_search_tool]

def chat(message, history):
    """
    Main chat function that handles user messages and tool calls.
    """
    # Build the conversation messages
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    
    # Make the initial API call
    response = openai.chat.completions.create(
        model=MODEL, 
        messages=messages, 
        tools=tools
    )
    
    # Check if the model wants to call a tool
    if response.choices[0].finish_reason == "tool_calls":
        # Get the assistant's message with tool calls
        assistant_message = response.choices[0].message
        
        # Add the assistant's message with tool calls to the conversation
        messages.append({
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in assistant_message.tool_calls
            ]
        })
        
        # Handle each tool call and add tool responses
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name == "get_flight_offers":
                # Call our flight search function
                result = get_flight_offers(**function_args)
                
                # Add the tool response message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, indent=2)
                })
        
        # Make another API call to get the final response
        final_response = openai.chat.completions.create(
            model=MODEL, 
            messages=messages
        )
        
        return final_response.choices[0].message.content
    
    # If no tool call, return the direct response
    return response.choices[0].message.content

gr.ChatInterface(fn=chat, type="messages").launch()

