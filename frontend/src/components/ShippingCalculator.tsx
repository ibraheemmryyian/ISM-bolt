import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { supabase } from '../lib/supabase';
import { 
  Truck, 
  Package, 
  DollarSign, 
  Clock, 
  MapPin,
  CheckCircle,
  AlertCircle,
  Loader2
} from 'lucide-react';

interface ShippingRate {
  id: string;
  carrier: string;
  service: string;
  rate: string;
  currency: string;
  delivery_days: number;
  tracking: boolean;
  insurance: boolean;
  material_id: string;
  seller_company: string;
}

interface ShippingCalculatorProps {
  materialId: string;
  materialName: string;
  sellerCompany: string;
  onRateSelect?: (rate: ShippingRate) => void;
  onExchangeCreate?: (exchangeData: any) => void;
}

const ShippingCalculator: React.FC<ShippingCalculatorProps> = ({
  materialId,
  materialName,
  sellerCompany,
  onRateSelect,
  onExchangeCreate
}) => {
  const [buyerLocation, setBuyerLocation] = useState({
    name: '',
    company: '',
    street1: '',
    city: '',
    state: '',
    zip: '',
    country: 'US',
    phone: '',
    email: ''
  });
  const [rates, setRates] = useState<ShippingRate[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedRate, setSelectedRate] = useState<ShippingRate | null>(null);
  const [error, setError] = useState<string | null>(null);

  const calculateRates = async () => {
    if (!buyerLocation.street1 || !buyerLocation.city || !buyerLocation.zip) {
      setError('Please fill in all required address fields');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/shipping/calculate-rates', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          materialId,
          buyerLocation
        })
      });

      const data = await response.json();

      if (data.success) {
        setRates(data.rates);
      } else {
        setError(data.error || 'Failed to calculate shipping rates');
      }
    } catch (error) {
      console.error('Error calculating rates:', error);
      setError('Failed to calculate shipping rates');
    } finally {
      setLoading(false);
    }
  };

  const handleRateSelect = (rate: ShippingRate) => {
    setSelectedRate(rate);
    onRateSelect?.(rate);
  };

  const createExchange = async () => {
    if (!selectedRate) {
      setError('Please select a shipping rate');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Get current user's company info
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error('Not authenticated');

      const { data: buyerCompany } = await supabase
        .from('companies')
        .select('*')
        .eq('id', user.id)
        .single();

      const exchangeData = {
        from_company_id: selectedRate.seller_company, // This should be the seller's company ID
        to_company_id: user.id,
        material_id: materialId,
        quantity: 1, // This should come from the material listing
        from_address: {
          name: sellerCompany,
          company: sellerCompany,
          street1: '123 Industrial Way', // This should come from seller's profile
          city: 'Manufacturing City',
          state: 'CA',
          zip: '90210',
          country: 'US',
          phone: '+1234567890',
          email: 'contact@company.com'
        },
        to_address: buyerLocation,
        package_details: {
          weight: 10,
          length: 12,
          width: 12,
          height: 12
        }
      };

      const response = await fetch('/api/shipping/create-exchange', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(exchangeData)
      });

      const data = await response.json();

      if (data.success) {
        onExchangeCreate?.(data.exchange);
        // Show success message
        alert('Material exchange created successfully! Check your shipping history for tracking details.');
      } else {
        setError(data.error || 'Failed to create exchange');
      }
    } catch (error) {
      console.error('Error creating exchange:', error);
      setError('Failed to create material exchange');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Truck className="h-5 w-5 text-blue-600" />
          Shipping Calculator
        </CardTitle>
        <p className="text-sm text-gray-600">
          Calculate shipping costs for {materialName} from {sellerCompany}
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Buyer Address Form */}
        <div className="space-y-4">
          <h3 className="font-medium text-gray-900">Your Shipping Address</h3>
          <div className="grid grid-cols-2 gap-4">
            <Input
              placeholder="Full Name"
              value={buyerLocation.name}
              onChange={(e) => setBuyerLocation(prev => ({ ...prev, name: e.target.value }))}
            />
            <Input
              placeholder="Company"
              value={buyerLocation.company}
              onChange={(e) => setBuyerLocation(prev => ({ ...prev, company: e.target.value }))}
            />
            <Input
              placeholder="Street Address"
              value={buyerLocation.street1}
              onChange={(e) => setBuyerLocation(prev => ({ ...prev, street1: e.target.value }))}
              className="col-span-2"
            />
            <Input
              placeholder="City"
              value={buyerLocation.city}
              onChange={(e) => setBuyerLocation(prev => ({ ...prev, city: e.target.value }))}
            />
            <Input
              placeholder="State"
              value={buyerLocation.state}
              onChange={(e) => setBuyerLocation(prev => ({ ...prev, state: e.target.value }))}
            />
            <Input
              placeholder="ZIP Code"
              value={buyerLocation.zip}
              onChange={(e) => setBuyerLocation(prev => ({ ...prev, zip: e.target.value }))}
            />
            <Input
              placeholder="Phone"
              value={buyerLocation.phone}
              onChange={(e) => setBuyerLocation(prev => ({ ...prev, phone: e.target.value }))}
            />
          </div>
          
          <Button 
            onClick={calculateRates}
            disabled={loading}
            className="w-full"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Calculating Rates...
              </>
            ) : (
              <>
                <Package className="h-4 w-4 mr-2" />
                Calculate Shipping Rates
              </>
            )}
          </Button>
        </div>

        {/* Error Message */}
        {error && (
          <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-lg">
            <AlertCircle className="h-4 w-4 text-red-600" />
            <span className="text-red-700 text-sm">{error}</span>
          </div>
        )}

        {/* Shipping Rates */}
        {rates.length > 0 && (
          <div className="space-y-4">
            <h3 className="font-medium text-gray-900">Available Shipping Options</h3>
            <div className="space-y-3">
              {rates.map((rate) => (
                <div
                  key={rate.id}
                  className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                    selectedRate?.id === rate.id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                  onClick={() => handleRateSelect(rate)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-2">
                        <Truck className="h-4 w-4 text-gray-600" />
                        <span className="font-medium">{rate.carrier}</span>
                      </div>
                      <Badge variant="outline">{rate.service}</Badge>
                    </div>
                    <div className="text-right">
                      <div className="font-bold text-lg">
                        <DollarSign className="inline h-4 w-4" />
                        {rate.rate}
                      </div>
                      <div className="flex items-center gap-1 text-sm text-gray-600">
                        <Clock className="h-3 w-3" />
                        {rate.delivery_days} days
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 mt-2 text-sm text-gray-600">
                    {rate.tracking && (
                      <span className="flex items-center gap-1">
                        <CheckCircle className="h-3 w-3 text-green-600" />
                        Tracking
                      </span>
                    )}
                    {rate.insurance && (
                      <span className="flex items-center gap-1">
                        <CheckCircle className="h-3 w-3 text-green-600" />
                        Insurance
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Create Exchange Button */}
            {selectedRate && (
              <Button 
                onClick={createExchange}
                disabled={loading}
                className="w-full"
                size="lg"
              >
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Creating Exchange...
                  </>
                ) : (
                  <>
                    <Package className="h-4 w-4 mr-2" />
                    Create Material Exchange
                  </>
                )}
              </Button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ShippingCalculator; 