import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';

export function TransactionPage() {
  const { connectionId } = useParams();
  const navigate = useNavigate();
  const [connection, setConnection] = useState<any>(null);
  const [material, setMaterial] = useState<any>(null);
  const [amount, setAmount] = useState<number>(0);
  const [fee, setFee] = useState<number>(0);
  const [status, setStatus] = useState<string>('pending');
  const feePercentage = 0.02; // 2%

  useEffect(() => {
    async function fetchConnection() {
      const { data } = await supabase
        .from('connections')
        .select('*')
        .eq('id', connectionId)
        .single();
      setConnection(data);
      // Optionally fetch related material
      // setMaterial(...)
    }
    fetchConnection();
  }, [connectionId]);

  function handleAmountChange(e: React.ChangeEvent<HTMLInputElement>) {
    const value = parseFloat(e.target.value);
    setAmount(value);
    setFee(value * feePercentage);
  }

  async function handleConfirm() {
    if (!connection) return;
    await supabase.from('transactions').insert({
      connection_id: connection.id,
      buyer_id: connection.requester_id,
      seller_id: connection.recipient_id,
      material_id: null, // Optionally link to a material
      amount,
      fee,
      fee_percentage: feePercentage,
      status: 'confirmed',
      created_at: new Date().toISOString()
    });
    setStatus('confirmed');
    // Optionally navigate or show a success message
    setTimeout(() => navigate('/dashboard'), 2000);
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="bg-white rounded-xl shadow-sm p-8 w-full max-w-lg">
        <h2 className="text-2xl font-bold mb-4">Transaction Details</h2>
        {connection ? (
          <>
            <div className="mb-4">
              <div><b>Buyer:</b> {connection.requester_id}</div>
              <div><b>Seller:</b> {connection.recipient_id}</div>
            </div>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">Transaction Amount</label>
              <input
                type="number"
                value={amount}
                onChange={handleAmountChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                min={0}
              />
            </div>
            <div className="mb-4">
              <div><b>Transaction Fee (2%):</b> {fee.toFixed(2)}</div>
              <div><b>Total:</b> {(amount + fee).toFixed(2)}</div>
            </div>
            <button
              onClick={handleConfirm}
              className="w-full bg-emerald-500 text-white py-2 px-4 rounded-lg hover:bg-emerald-600 transition"
              disabled={status === 'confirmed'}
            >
              {status === 'confirmed' ? 'Confirmed!' : 'Confirm Transaction'}
            </button>
          </>
        ) : (
          <div>Loading...</div>
        )}
      </div>
    </div>
  );
} 